# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""문항 단위 부트스트랩 — Dev 880 자체가 한 번의 추출이라는 오차를 잰다.

`final_judgment.py` 의 분할 반복은 "어느 절반을 보정셋으로 삼았나"에서 오는 잡음만
줄인다. 분할을 200회로 늘려 표준오차가 0.016까지 내려가도, 그 숫자는 **이 880 위에서**
방식을 비교할 때의 해상도이지 채점셋 점수의 정밀도가 아니다. 채점셋은 다른 문항
880개이기 때문이다.

여기서는 Dev 880을 복원추출해 "다른 880이었다면" 을 B회 모사하고, 각 복제표본 안에서
분할 k회를 돌려 방식별 점수를 낸다. 모든 방식이 **같은 복제표본·같은 분할**을 보므로
방식 간 비교는 쌍대로 이뤄진다.

## 분산 분해

복제표본별 평균의 분산 V_b 는 두 성분을 합친 값이다.

    V_b = 문항추출 분산 + 분할 분산 / k

복제표본 안의 분할 간 분산 V_w 가 곧 분할 분산이므로,

    문항추출 분산 ≈ V_b − V_w / k

분할을 아무리 늘려도 첫 항은 줄지 않는다. 이 값이 "채점셋에서 얼마나 흔들리는가" 의
하한이다.

## 중복 문항 처리

복원추출은 같은 문항을 여러 번 뽑는다. id 를 그대로 두면 채점기의 decision 색인
(episode_id -> model_id) 에서 중복이 하나로 접혀, 배분기가 두 사본을 다르게 골랐을 때
예산 회계가 어긋난다. 그래서 사본마다 새 id 를 주고 outcome 행도 함께 복제한다.

## 사용

    .venv-train/bin/python analysis/episode_bootstrap.py \\
        --replicates 100 --splits-per-replicate 4 \\
        --report analysis/reports/episode-bootstrap.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import statistics
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.final_judgment import (  # noqa: E402
    TIER_WEIGHT,
    assemble_predictions,
    build_cost_estimators,
)
from analysis.holdout_protocol import (  # noqa: E402
    apply_policy,
    score_tier,
    select_safety,
    subset,
)
from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

_CTX: Dict[str, Any] = {}


def resample_episodes(
    inputs: Any, outcomes: Any, seed: int
) -> Tuple[Any, Any, List[int]]:
    """문항을 복원추출한다. 사본에는 새 id 를 주고 outcome 행도 복제한다."""

    count = len(inputs.episodes)
    rng = random.Random(seed)
    picks = rng.choices(range(count), k=count)

    rows_by_id: Dict[str, List[Any]] = {}
    for row in outcomes.outcomes:
        rows_by_id.setdefault(row.episode_id, []).append(row)

    episodes: List[Any] = []
    rows: List[Any] = []
    seen: Dict[int, int] = {}
    for source in picks:
        episode = inputs.episodes[source]
        copy_index = seen.get(source, 0)
        seen[source] = copy_index + 1
        new_id = (
            episode.episode_id
            if copy_index == 0
            else f"{episode.episode_id}#b{copy_index}"
        )
        episodes.append(dataclasses.replace(episode, episode_id=new_id))
        for row in rows_by_id[episode.episode_id]:
            rows.append(dataclasses.replace(row, episode_id=new_id))

    return (
        dataclasses.replace(inputs, episodes=tuple(episodes)),
        dataclasses.replace(outcomes, outcomes=tuple(rows)),
        picks,
    )


def evaluate_replicate(replicate: int) -> Tuple[int, Dict[str, List[float]]]:
    """복제표본 하나에서 모든 방식을 같은 분할로 채점한다.

    같은 복제표본·같은 분할을 공유해야 방식 간 차이에서 '이 표본이 어려웠다' 성분이
    상쇄된다. 그래서 분할은 방식 루프 바깥에서 한 번만 만든다.
    """

    base_inputs = _CTX["inputs"]
    base_outcomes = _CTX["outcomes"]
    policy = _CTX["policy"]
    grid_size = _CTX["grid_size"]
    splits_per = _CTX["splits_per_replicate"]
    seed = _CTX["seed"]

    inputs, outcomes, picks = resample_episodes(
        base_inputs, base_outcomes, seed + replicate
    )
    count = len(picks)
    half = count // 2

    rng = random.Random(seed + 1_000_000 + replicate)
    splits: List[Tuple[List[int], List[int]]] = []
    for _ in range(splits_per):
        order = list(range(count))
        rng.shuffle(order)
        splits.append((order[:half], order[half:]))

    out: Dict[str, List[float]] = {}
    for name, (base_predictions, safety_policy) in _CTX["methods"].items():
        predictions = [base_predictions[i] for i in picks]
        scores: List[float] = []
        for calib_index, holdout_index in splits:
            calib = subset(inputs, outcomes, calib_index)
            holdout = subset(inputs, outcomes, holdout_index)
            calib_pred = [predictions[i] for i in calib_index]
            holdout_pred = [predictions[i] for i in holdout_index]
            weighted = Decimal(0)
            for tier in TIERS:
                if safety_policy == "trust":
                    safety = 1.0
                else:
                    safety, _report = select_safety(
                        calib[0], calib[1], policy, calib_pred, tier, grid_size
                    )
                selected = apply_policy(holdout_pred, policy, tier, safety)
                row = score_tier(
                    holdout[0], holdout[1], policy, tier, selected
                )
                raw = float(Decimal(row["tier_score"]))
                # 초과 시 0 — 실제 채점 규칙
                effective = raw if bool(row["budget_passed"]) else 0.0
                weighted += TIER_WEIGHT[tier] * Decimal(str(effective))
            scores.append(float(weighted))
        out[name] = scores
    return replicate, out


def decompose(
    per_replicate: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """복제표본별 k개 값에서 문항추출 성분과 분할 성분을 분리한다."""

    means = [statistics.fmean(scores) for scores in per_replicate]
    splits_per = len(per_replicate[0])
    between = statistics.variance(means) if len(means) > 1 else 0.0
    within_parts = [
        statistics.variance(scores) for scores in per_replicate if len(scores) > 1
    ]
    within = statistics.fmean(within_parts) if within_parts else 0.0
    # V_b = 문항추출 분산 + 분할 분산 / k  →  문항추출 분산 = V_b − V_w/k
    episode_var = between - within / splits_per
    return {
        "mean": statistics.fmean(means),
        "between_replicate_var": between,
        "within_replicate_var": within,
        # 음수는 추정 잡음 — 문항추출 성분이 분할 성분에 묻혔다는 뜻
        "episode_var": episode_var,
        "episode_stdev": math.sqrt(episode_var) if episode_var > 0 else 0.0,
        "replicates": len(means),
        "splits_per_replicate": splits_per,
    }


def detection_floor(within_var: float, replicates: int, splits_per: int) -> float:
    """이 설정으로 탐지 가능한 최소 문항추출 SD.

    문항추출 분산은 V_b − V_w/k 로 얻는데, V_b 자체의 추정 오차가
    V_b·√(2/(B−1)) 이다. 그 오차의 2배보다 커야 0과 구분되므로

        e² > 2·(e² + s²/k)·√(2/(B−1))

    를 e² 에 대해 풀면 하한이 나온다. 이 값보다 작게 나온 추정치는
    '문항추출 성분이 없다'가 아니라 '이 설정으로는 못 잰다'는 뜻이다.
    """

    if replicates < 3 or splits_per < 1:
        return float("inf")
    relative = math.sqrt(2.0 / (replicates - 1))
    denominator = 1.0 - 2.0 * relative
    if denominator <= 0:
        return float("inf")
    floor_var = 2.0 * relative * (within_var / splits_per) / denominator
    return math.sqrt(max(floor_var, 0.0))


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(
    collected: Sequence[Mapping[str, List[float]]], method_names: Sequence[str]
) -> Dict[str, Any]:
    by_method = {
        name: [replicate[name] for replicate in collected] for name in method_names
    }
    means = {
        name: [statistics.fmean(scores) for scores in runs]
        for name, runs in by_method.items()
    }

    # 승률 — 복제표본마다 1위를 세면 '순위가 표본에 따라 흔들리는 정도'가 그대로 나온다.
    wins = {name: 0 for name in method_names}
    for index in range(len(collected)):
        best = max(method_names, key=lambda n: means[n][index])
        wins[best] += 1

    methods: Dict[str, Any] = {}
    for name in method_names:
        stats = decompose(by_method[name])
        stats["ci_low"] = percentile(means[name], 0.025)
        stats["ci_high"] = percentile(means[name], 0.975)
        stats["win_rate"] = wins[name] / len(collected)
        stats["replicate_means"] = means[name]
        methods[name] = stats

    reference = max(method_names, key=lambda n: methods[n]["mean"])
    pairs = []
    for name in method_names:
        if name == reference:
            continue
        diffs = [
            [a - b for a, b in zip(ref_scores, other_scores)]
            for ref_scores, other_scores in zip(
                by_method[reference], by_method[name]
            )
        ]
        stats = decompose(diffs)
        diff_means = [statistics.fmean(d) for d in diffs]
        pairs.append(
            {
                "reference": reference,
                "method": name,
                "mean_difference": stats["mean"],
                "episode_stdev": stats["episode_stdev"],
                "within_replicate_var": stats["within_replicate_var"],
                "ci_low": percentile(diff_means, 0.025),
                "ci_high": percentile(diff_means, 0.975),
                # 부호가 뒤집히는 복제표본 비율 — 순위 역전 확률
                "reversal_rate": statistics.fmean(
                    1.0 if value <= 0 else 0.0 for value in diff_means
                ),
            }
        )
    return {
        "report_type": "episode-bootstrap-v1",
        "reference": reference,
        "methods": methods,
        "pairs": pairs,
    }


def report(summary: Mapping[str, Any]) -> None:
    methods = summary["methods"]
    order = sorted(methods, key=lambda n: -methods[n]["mean"])
    width = max(len(name) for name in order) + 2
    first = methods[order[0]]

    print(
        f"\n[B1] 문항 부트스트랩 — 복제표본 {first['replicates']}회"
        f" × 분할 {first['splits_per_replicate']}회"
    )
    print(
        f"  {'방식':<{width}}{'평균':>9}{'95% 구간':>19}"
        f"{'문항추출SD':>12}{'승률':>8}"
    )
    for name in order:
        stats = methods[name]
        span = f"[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
        print(
            f"  {name:<{width}}{stats['mean']:>9.4f}{span:>19}"
            f"{stats['episode_stdev']:>12.4f}{stats['win_rate']*100:>7.0f}%"
        )

    floor = detection_floor(
        statistics.fmean(methods[n]["within_replicate_var"] for n in order),
        first["replicates"],
        first["splits_per_replicate"],
    )
    print(
        f"     이 설정의 탐지 하한: 문항추출 SD {floor:.4f}"
        " — 이보다 작은 값은 '없다'가 아니라 '못 잰다'"
    )

    print("\n[B2] 분산 분해 — 분할을 늘려도 줄지 않는 성분은 왼쪽")
    print(
        f"  {'방식':<{width}}{'문항추출SD':>12}{'분할SD':>10}"
        f"{'문항추출 비중':>14}"
    )
    for name in order:
        stats = methods[name]
        split_sd = math.sqrt(max(stats["within_replicate_var"], 0.0))
        total = stats["episode_var"] + stats["within_replicate_var"]
        share = stats["episode_var"] / total if total > 0 else 0.0
        print(
            f"  {name:<{width}}{stats['episode_stdev']:>12.4f}"
            f"{split_sd:>10.4f}{max(share, 0.0)*100:>13.0f}%"
        )

    pair_within = statistics.fmean(
        row["within_replicate_var"] for row in summary["pairs"]
    )
    pair_floor = detection_floor(
        pair_within, first["replicates"], first["splits_per_replicate"]
    )
    print(f"\n[B3] 쌍대 차이의 표본 불확실성 (기준: {summary['reference']})")
    print(f"     쌍대 탐지 하한: 문항추출 SD {pair_floor:.4f}")
    print("     역전율 = 다른 880을 뽑았을 때 기준 방식이 지는 복제표본 비율")
    print(
        f"  {'방식':<{width}}{'평균차':>9}{'95% 구간':>19}{'역전율':>9}"
    )
    for row in sorted(summary["pairs"], key=lambda r: r["mean_difference"]):
        span = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        print(
            f"  {row['method']:<{width}}{row['mean_difference']:>9.4f}"
            f"{span:>19}{row['reversal_rate']*100:>8.0f}%"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="episode-bootstrap",
        description="문항 복원추출로 Dev 880 자체의 추출 오차를 측정합니다.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "baselines" / "hash-regex-public.v1.json",
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--l2", type=float, default=100.0 / 1760.0)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=4.0)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--splits-per-replicate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = hash_regex.load_artifact(args.artifact)
    policy = load_bundled_policy()
    inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")

    print("=" * 78)
    print("문항 단위 부트스트랩 — Dev 880 추출 오차")
    print(
        f"  복제표본 {args.replicates}회 × 분할 {args.splits_per_replicate}회"
        f" · 안전계수 그리드 {args.grid_size}"
    )
    print("=" * 78)

    print("  비용 추정기 적합 중...")
    estimators = build_cost_estimators(
        artifact, args.tau, l2=args.l2, iterations=args.iterations,
        learning_rate=args.learning_rate, seed=args.seed,
    )

    specs: List[Tuple[str, Any, str]] = [("shipped", None, "maxscore")]
    for name in ("ridge_log_exp", "ridge_smeared"):
        specs.append((name, estimators[name], "maxscore"))
    for tau in args.tau:
        key = f"quantile_{tau:g}"
        specs.append((key, estimators[key], "maxscore"))
        specs.append((f"{key}+trust", estimators[key], "trust"))
    specs.append(("ridge_smeared+trust", estimators["ridge_smeared"], "trust"))

    print("  예측 조립 중...")
    _CTX["inputs"] = inputs
    _CTX["outcomes"] = outcomes
    _CTX["policy"] = policy
    _CTX["grid_size"] = args.grid_size
    _CTX["splits_per_replicate"] = args.splits_per_replicate
    _CTX["seed"] = args.seed
    _CTX["methods"] = {
        name: (assemble_predictions(inputs.episodes, artifact, costs), safety)
        for name, costs, safety in specs
    }
    method_names = [name for name, _costs, _safety in specs]

    jobs = max(1, args.jobs)
    print(f"  복제표본 채점 중: {args.replicates}개 (병렬 {jobs})")
    collected: List[Any] = [None] * args.replicates
    if jobs > 1:
        import multiprocessing

        # fork 로 _CTX 를 물려준다 — spawn 이면 자식이 부모 전역을 못 본다.
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=jobs) as pool:
            done = 0
            for index, scores in pool.imap_unordered(
                evaluate_replicate, range(args.replicates), chunksize=1
            ):
                collected[index] = scores
                done += 1
                print(f"    {done}/{args.replicates}", flush=True)
    else:
        for index in range(args.replicates):
            _index, scores = evaluate_replicate(index)
            collected[index] = scores
            print(f"    {index + 1}/{args.replicates}", flush=True)

    summary = summarize(collected, method_names)
    report(summary)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.report}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
