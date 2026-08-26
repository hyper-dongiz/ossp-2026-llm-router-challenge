# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""안전계수 보정의 일반화 갭을 측정한다.

공개 baseline은 안전계수를 Dev로 고르고 성능도 Dev로 보고한다. 안전계수에 대해
in-sample이므로 "이 비용비율이 미지의 채점셋에서 얼마나 올라가는가"를 답할 수 없다.
채점셋에서 Premium이 3.985 -> 약 4.2로 올라가 0점이 된 것이 그 결과다.

이 스크립트는 Dev를 보정셋/홀드아웃으로 쪼개, 보정셋에서 고른 안전계수를 홀드아웃에서
측정한다. 두 값의 차이가 곧 일반화 갭이다. 갭을 두 성분으로 분리한다.

  selection  안전계수를 점수 최대화로 고르면서 생기는 편향 (보정셋 -> 홀드아웃)
  sampling   정책을 고정해도 문항 표본이 달라지면 생기는 변동 (부분집합 재추출)

회귀계수는 공개 artifact를 그대로 쓴다. 계수는 Train에서만 학습돼 Dev를 본 적이
없으므로, Dev 부분집합은 계수에 대해 여전히 홀드아웃이다. 실패한 단계인 안전계수
선택만 재현하면 되므로 재학습(numpy)이 필요 없다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import hash_regex  # noqa: E402
from ossp_router.heuristic import episode_text, extract_features  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    Decision,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from ossp_router.scoring import score_submissions  # noqa: E402

Prediction = Tuple[Mapping[str, float], Mapping[str, float]]


def subset(
    inputs: InputBatch, outcomes: OutcomeBatch, indices: Sequence[int]
) -> Tuple[InputBatch, OutcomeBatch]:
    """문항 인덱스로 입력·평가결과를 함께 잘라낸다."""

    episodes = tuple(inputs.episodes[index] for index in indices)
    keep = {episode.episode_id for episode in episodes}
    return (
        dataclasses.replace(inputs, episodes=episodes),
        dataclasses.replace(
            outcomes,
            outcomes=tuple(o for o in outcomes.outcomes if o.episode_id in keep),
        ),
    )


def score_tier(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    tier: str,
    selected: Sequence[str],
) -> Mapping[str, Any]:
    """공식 채점기로 한 등급만 채점한다 (train_hash_regex._score_one_tier와 동일)."""

    all_light = tuple(policy.light_model_id for _ in inputs.episodes)
    submissions = [
        Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=policy.policy_id,
            split=inputs.split,
            tier=candidate,
            decisions=tuple(
                Decision(episode.episode_id, model_id)
                for episode, model_id in zip(
                    inputs.episodes, selected if candidate == tier else all_light
                )
            ),
        )
        for candidate in TIERS
    ]
    return score_submissions(inputs, outcomes, submissions, policy)["tiers"][tier]


def apply_policy(
    predictions: Sequence[Prediction],
    policy: RoutingPolicy,
    tier: str,
    safety: float,
) -> Tuple[str, ...]:
    """한 등급의 모델 선택을 만든다 (premium은 AX31 fill까지 baseline과 동일)."""

    scores = [item[0] for item in predictions]
    costs = [item[1] for item in predictions]
    multiplier = float(policy.tiers[tier].budget_multiplier)
    selected, _ratio = hash_regex.select_models(
        scores, costs, budget_multiplier=multiplier, safety_ratio=safety
    )
    if tier == "premium":
        selected, _ratio = hash_regex.fill_ax31_upgrades(
            selected,
            scores,
            costs,
            budget_multiplier=multiplier,
            safety_ratio=hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO,
        )
    return selected


def select_safety(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    predictions: Sequence[Prediction],
    tier: str,
    grid_size: int,
) -> Tuple[float, Mapping[str, Any]]:
    """보정셋에서 안전계수를 고른다.

    랭킹은 train_hash_regex._select_safety_ratios와 같다 — 점수 최대화가 1순위,
    비용비율은 동점처리용. 실패한 절차를 그대로 재현해야 갭이 측정되므로 바꾸지 않는다.
    """

    best: Any = None
    for safety in _safety_grid(policy, tier, grid_size):
        selected = apply_policy(predictions, policy, tier, safety)
        report = score_tier(inputs, outcomes, policy, tier, selected)
        rank = (
            Decimal(report["tier_score"]),
            -Decimal(report["budget_ratio"]),
            -Decimal(str(safety)),
        )
        if best is None or rank > best[0]:
            best = (rank, safety, report)
    assert best is not None
    return best[1], best[2]


def _safety_grid(policy: RoutingPolicy, tier: str, size: int) -> Tuple[float, ...]:
    minimum = 1.0 / float(policy.tiers[tier].budget_multiplier)
    if size <= 1 or minimum >= 1.0:
        return (min(1.0, minimum),)
    return tuple(
        minimum + (1.0 - minimum) * index / (size - 1) for index in range(size)
    )


Split = Tuple[List[int], List[int]]


def train_vocabulary(path: Path) -> set:
    """Train 문항에 등장한 토큰 집합 — 미등장 비율 계산용."""

    vocab = set()
    for episode in load_input(path).episodes:
        vocab.update(hash_regex._normalized_tokens(episode_text(episode)))
    return vocab


def _oov_ratio(episode: Any, vocab: set) -> float:
    tokens = hash_regex._normalized_tokens(episode_text(episode))
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token not in vocab) / len(tokens)


def build_splits(
    inputs: InputBatch,
    *,
    repeats: int,
    seed: int,
    vocab: set | None,
) -> Dict[str, List[Split]]:
    """분할 전략별 (보정셋, 홀드아웃) 목록.

    random 은 두 쪽의 문항 구성이 같아 '구성이 바뀔 때의 위험'을 과소평가한다.
    나머지는 특정 축으로 정렬해 반씩 가르므로 두 쪽의 구성이 의도적으로 다르다 —
    비공개 채점셋이 공개 자료와 다른 구성일 경우를 모사한다. 양방향 모두 본다.
    """

    count = len(inputs.episodes)
    half = count // 2
    plans: Dict[str, List[Split]] = {}

    plans["random"] = []
    for trial in range(repeats):
        order = list(range(count))
        random.Random(seed + trial).shuffle(order)
        plans["random"].append((order[:half], order[half:]))

    axes: Dict[str, Any] = {
        "length": lambda index: extract_features(inputs.episodes[index]).character_count,
        "hangul": lambda index: extract_features(inputs.episodes[index]).hangul_ratio,
        "numeric": lambda index: extract_features(inputs.episodes[index]).numeric_density,
    }
    if vocab is not None:
        axes["oov"] = lambda index: _oov_ratio(inputs.episodes[index], vocab)

    for name, key in axes.items():
        order = sorted(range(count), key=key)
        low, high = order[:half], order[half:]
        # 보정셋에 없는 종류를 홀드아웃에 몰아준다 (그리고 그 반대도)
        plans[f"{name}:low→high"] = [(low, high)]
        plans[f"{name}:high→low"] = [(high, low)]
    return plans


def run_selection_trials(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    predictions: Sequence[Prediction],
    plans: Sequence[Split],
    *,
    grid_size: int,
) -> Dict[str, List[Mapping[str, Any]]]:
    """보정셋에서 고르고 홀드아웃에서 측정한다 — selection 성분."""

    results: Dict[str, List[Mapping[str, Any]]] = {tier: [] for tier in TIERS}
    for trial, (calib_index, holdout_index) in enumerate(plans):
        calib = subset(inputs, outcomes, calib_index)
        holdout = subset(inputs, outcomes, holdout_index)
        calib_pred = [predictions[i] for i in calib_index]
        holdout_pred = [predictions[i] for i in holdout_index]
        for tier in TIERS:
            safety, calib_report = select_safety(
                calib[0], calib[1], policy, calib_pred, tier, grid_size
            )
            selected = apply_policy(holdout_pred, policy, tier, safety)
            holdout_report = score_tier(
                holdout[0], holdout[1], policy, tier, selected
            )
            calib_ratio = Decimal(calib_report["budget_ratio"])
            holdout_ratio = Decimal(holdout_report["budget_ratio"])
            results[tier].append(
                {
                    "trial": trial,
                    "safety_ratio": safety,
                    "calib_budget_ratio": float(calib_ratio),
                    "holdout_budget_ratio": float(holdout_ratio),
                    "drift": float(holdout_ratio - calib_ratio),
                    "holdout_budget_passed": holdout_report["budget_passed"],
                    "calib_tier_score": float(Decimal(calib_report["tier_score"])),
                    "holdout_tier_score": float(
                        Decimal(holdout_report["tier_score"])
                    ),
                }
            )
    return results


def run_fixed_policy_trials(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    predictions: Sequence[Prediction],
    shipped_safety: Mapping[str, float],
    *,
    repeats: int,
    seed: int,
) -> Dict[str, List[float]]:
    """공개 artifact의 안전계수를 고정한 채 부분집합만 바꾼다 — sampling 성분."""

    count = len(inputs.episodes)
    half = count // 2
    results: Dict[str, List[float]] = {tier: [] for tier in TIERS}
    for trial in range(repeats):
        order = list(range(count))
        random.Random(seed + 10_000 + trial).shuffle(order)
        index = order[:half]
        part = subset(inputs, outcomes, index)
        part_pred = [predictions[i] for i in index]
        for tier in TIERS:
            selected = apply_policy(
                part_pred, policy, tier, shipped_safety[tier]
            )
            report = score_tier(part[0], part[1], policy, tier, selected)
            results[tier].append(float(Decimal(report["budget_ratio"])))
    return results


def summarize(values: Sequence[float]) -> Mapping[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holdout-protocol",
        description="안전계수 보정의 일반화 갭을 측정합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--train-input",
        type=Path,
        help="미등장 토큰 비율 축을 쓰려면 Train 입력 경로 (없으면 oov 분할 생략)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    policy = load_bundled_policy()
    artifact = hash_regex.load_artifact(args.artifact)

    predictions = [
        hash_regex.predict_episode(episode, artifact)
        for episode in inputs.episodes
    ]

    vocab = train_vocabulary(args.train_input) if args.train_input else None
    plans = build_splits(
        inputs, repeats=args.repeats, seed=args.seed, vocab=vocab
    )

    report: Dict[str, Any] = {
        "report_type": "holdout-generalization-gap-v2",
        "num_episodes": len(inputs.episodes),
        "subset_size": len(inputs.episodes) // 2,
        "repeats": args.repeats,
        "grid_size": args.grid_size,
        "seed": args.seed,
        "policy_id": policy.policy_id,
        "shipped_safety_ratios": dict(artifact.tier_safety_ratios),
        "strategies": {},
    }

    for name, split_plans in plans.items():
        selection = run_selection_trials(
            inputs, outcomes, policy, predictions, split_plans,
            grid_size=args.grid_size,
        )
        block: Dict[str, Any] = {"num_trials": len(split_plans), "tiers": {}}
        for tier in TIERS:
            rows = selection[tier]
            failures = [row for row in rows if not row["holdout_budget_passed"]]
            block["tiers"][tier] = {
                "budget_multiplier": float(policy.tiers[tier].budget_multiplier),
                "calib_budget_ratio": summarize(
                    [row["calib_budget_ratio"] for row in rows]
                ),
                "holdout_budget_ratio": summarize(
                    [row["holdout_budget_ratio"] for row in rows]
                ),
                "drift": summarize([row["drift"] for row in rows]),
                "holdout_fail_count": len(failures),
                "holdout_fail_rate": len(failures) / len(rows),
                "chosen_safety_ratio": summarize(
                    [row["safety_ratio"] for row in rows]
                ),
                "trials": rows,
            }
        report["strategies"][name] = block

    sampling = run_fixed_policy_trials(
        inputs, outcomes, policy, predictions, artifact.tier_safety_ratios,
        repeats=args.repeats, seed=args.seed,
    )
    report["sampling_fixed_policy"] = {
        tier: {
            "budget_ratio": summarize(sampling[tier]),
            "over_limit_count": sum(
                1 for value in sampling[tier]
                if value > float(policy.tiers[tier].budget_multiplier)
            ),
        }
        for tier in TIERS
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(f"OK: {args.report}\n")
    print(f"{'분할 전략':<20}{'등급':<10}{'보정':>9}{'홀드아웃':>10}{'한도':>6}{'실패':>8}")
    for name, block in report["strategies"].items():
        for tier in TIERS:
            t = block["tiers"][tier]
            flag = "  ←초과" if t["holdout_fail_count"] else ""
            print(
                f"{name:<20}{tier:<10}"
                f"{t['calib_budget_ratio']['median']:>9.4f}"
                f"{t['holdout_budget_ratio']['median']:>10.4f}"
                f"{t['budget_multiplier']:>6}"
                f"{t['holdout_fail_count']:>5}/{block['num_trials']}{flag}"
            )
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
