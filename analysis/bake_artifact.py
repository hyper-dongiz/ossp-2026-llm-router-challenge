# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""최종 정책을 제출용 아티팩트로 굽는다.

`docs/EVALUATION.md` §7 의 정책을 공개 hash-regex 와 **동일한 아티팩트 스키마**에
담는다. 스키마가 같으므로 추론 코드를 새로 쓸 필요가 없다 — 컨테이너는
`baselines/hash_regex.py` 를 그대로 쓰고 계수만 우리 것으로 바뀐다.

## 세 부분이 어떻게 스키마에 들어가는가

    점수 헤드     ridge(alpha=1000) 계수를 score_heads 에 그대로
    비용 헤드     ridge(log-cost) + Duan smearing. 추론이 exp(선형) 이므로
                  smearing 계수 f 를 절편에 접는다: exp(b)·f = exp(b + ln f)
    안전계수      tier_safety_ratios 에 박는다 (아래 참조)

## 안전계수를 왜 구워야 하는가

우리 규칙은 보정셋에서 안전계수를 고른다. 그런데 추론 시점에는 outcomes 가 없다 —
프롬프트만 들어온다. 따라서 오프라인에서 정하는 수밖에 없고, 가진 보정 데이터 전부
(공개 Dev 880)를 써서 정한다.

측정할 때는 440 으로 보정하고 나머지 440 에서 채점했다. 배포에서는 보정 데이터가
두 배이므로 그만큼 유리한 방향이다 — 다만 **이 아티팩트로 Dev 를 채점한 수치는
in-sample 이라 보고에 쓰면 안 된다.** 성능 근거는 홀드아웃 프로토콜 쪽 수치다.

## 사용

    .venv-train/bin/python analysis/bake_artifact.py \\
        --out baselines/hyper-dongiz.v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.fast_score import outcome_table  # noqa: E402
from analysis.final_judgment import assemble_predictions  # noqa: E402
from analysis.quantile_cost import MODELS, build_matrix  # noqa: E402
from analysis.risk_safety import select_safety_risk  # noqa: E402
from analysis.score_heads import build_multi, fit_ridge  # noqa: E402
from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

# EVALUATION.md §7 의 확정 값. 바꾸려면 문서와 log.md 를 함께 고칠 것.
#
# fit_ridge 는 데이터항을 행 수로 나누므로 l2 = alpha/rows 여야 실효 alpha 가 고정된다.
# 학습 split 을 늘리면 행 수가 변하므로 상수로 두면 정규화 강도가 밀린다.
SCORE_ALPHA = 1000.0
COST_ALPHA = 100.0
# 모델별 예측 수축 (light : ax31 : think). light 는 880 문항 전부의 기준점이라 그
# 변동이 모든 승급 판단에 잡음으로 들어간다 — 눌러야 한다. 승급 후보들은 그 변동이
# 곧 선택 신호라 건드리지 않는다. 라운드 13·15·18 과 다섯 목적함수에서 재현됨.
SCORE_SHRINK = {"ax31-light": 0.85, "ax31": 1.0, "axk1-think": 1.0}
SAFETY_RULE = "expected1.41"
GRID_SIZE = 25


def fit_score_heads(
    artifact: Any, splits: Sequence[str] = ("train",)
) -> Dict[str, Dict[str, Any]]:
    """ridge(alpha=SCORE_ALPHA) 점수 헤드. 절편은 0번 열."""

    xtr, ytr, _ttr = build_multi(splits, artifact)
    heads = {}
    for model_id in MODELS:
        weights = fit_ridge(xtr, ytr[model_id], l2=SCORE_ALPHA / xtr.shape[0])
        # 수축을 계수로 접는다: p' = m + g(b + x·w − m) = [g·b + (1−g)m] + g(x·w).
        # 중심 m 은 학습셋 예측 평균 — 배포 시점에 채점셋 평균은 알 수 없다.
        gamma = SCORE_SHRINK[model_id]
        centre = float((xtr @ weights).mean())
        folded = weights * gamma
        folded[0] = gamma * weights[0] + (1.0 - gamma) * centre
        heads[model_id] = {
            "intercept": float(folded[0]),
            "coefficients": [float(v) for v in folded[1:]],
        }
    return heads


def fit_cost_heads(
    artifact: Any, splits: Sequence[str] = ("train",)
) -> Dict[str, Dict[str, Any]]:
    """log-cost ridge + Duan smearing.

    smearing 계수는 Train 에서 실측합/예측합이 아니라 비율의 평균으로 낸다
    (`final_judgment.build_cost_estimators` 와 동일). 추론이 exp(선형) 이므로
    ln(계수) 를 절편에 더해 접는다.
    """

    import numpy

    mats, targets = [], None
    for split in splits:
        matrix, cost, _ids = build_matrix(split, artifact)
        mats.append(matrix)
        if targets is None:
            targets = {m: [cost[m]] for m in cost}
        else:
            for m in cost:
                targets[m].append(cost[m])
    xtr = numpy.vstack(mats)
    ytr = {m: numpy.concatenate(v) for m, v in targets.items()}
    heads = {}
    for model_id in MODELS:
        target = numpy.log(numpy.maximum(ytr[model_id], 1e-12))
        weights = fit_ridge(xtr, target, l2=COST_ALPHA / xtr.shape[0])
        predicted = numpy.exp(numpy.clip(xtr @ weights, -50.0, 50.0))
        factor = float(numpy.mean(ytr[model_id] / numpy.maximum(predicted, 1e-12)))
        heads[model_id] = {
            "intercept": float(weights[0]) + math.log(factor),
            "coefficients": [float(v) for v in weights[1:]],
            "_smearing_factor": factor,  # 기록용. 아래에서 제거한다
        }
    return heads


def choose_safety(
    artifact: Any, score_heads: Dict[str, Any], cost_heads: Dict[str, Any]
) -> Dict[str, float]:
    """공개 Dev 880 전체를 보정셋으로 삼아 등급별 안전계수를 고른다."""

    policy = load_bundled_policy()
    inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")

    # 구운 계수를 그대로 쓴 예측 — 배포 시점과 같은 입력이어야 안전계수가 맞는다.
    staged = _clone_with_heads(artifact, score_heads, cost_heads)
    predictions = assemble_predictions(inputs.episodes, staged, None)
    table = outcome_table(inputs, outcomes, policy)

    chosen = {}
    for tier in TIERS:
        safety, stats = select_safety_risk(
            predictions, policy, tier, GRID_SIZE, table, SAFETY_RULE
        )
        chosen[tier] = (safety, stats)
    return chosen


def _clone_with_heads(
    artifact: Any, score_heads: Dict[str, Any], cost_heads: Dict[str, Any]
) -> Any:
    """계수만 바꾼 아티팩트 객체 (안전계수는 나중에 정해지므로 원본 값 유지)."""

    payload = _artifact_dict(artifact, score_heads, cost_heads, artifact.tier_safety_ratios)
    return hash_regex.parse_artifact(payload)


def _artifact_dict(
    artifact: Any,
    score_heads: Dict[str, Any],
    cost_heads: Dict[str, Any],
    safety: Any,
) -> Dict[str, Any]:
    """공개 아티팩트와 **정확히 같은 키 집합**으로 만든다 (parse_artifact 가 엄격)."""

    source = json.loads(
        (ROOT / "baselines" / "hash-regex-public.v1.json").read_text("utf-8")
    )
    clean_cost = {
        m: {k: v for k, v in head.items() if not k.startswith("_")}
        for m, head in cost_heads.items()
    }
    source["score_heads"] = score_heads
    source["log_cost_heads"] = clean_cost
    source["tier_safety_ratios"] = {t: float(safety[t]) for t in TIERS}
    return source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bake-artifact", description="최종 정책을 제출용 아티팩트로 굽습니다."
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "baselines" / "hyper-dongiz.v1.json"
    )
    parser.add_argument(
        "--fit-splits", nargs="+", default=["train"], choices=["train", "dev"],
        help="헤드를 적합할 split. 안전계수는 항상 Train 전용 헤드로 고른다",
    )
    args = parser.parse_args(argv)

    base = hash_regex.load_artifact(ROOT / "baselines" / "hash-regex-public.v1.json")

    print("=" * 74)
    print("최종 정책 아티팩트 굽기")
    print(f"  점수 헤드  ridge(alpha={SCORE_ALPHA:g}) + 수축 "
          + "/".join(f"{SCORE_SHRINK[m]:g}" for m in MODELS))
    print(f"  비용 헤드  ridge(alpha={COST_ALPHA:g}) + Duan smearing (절편에 접음)")
    print(f"  안전계수   {SAFETY_RULE} 규칙, 공개 Dev 880 전체를 보정셋으로")
    print(f"  헤드 학습  {'+'.join(args.fit_splits)}")
    print("=" * 74)

    # 안전계수는 **항상 Train 전용 헤드로** 고른다.
    #
    # Dev 까지 학습에 넣으면 Dev 예측이 in-sample 이 되어 오차가 실제보다 작아 보이고,
    # 보정이 그만큼 공격적인 안전계수를 고른다. 그러면 채점셋에서 예산을 넘긴다 —
    # baseline 이 저지른 것과 같은 구조다.
    #
    # 그래서 보정은 정직한(Train 전용) 헤드로 하고, 제출에는 더 좋은 헤드를 싣는다.
    # 헤드가 좋아지면 예측-실측 괴리가 줄어들 뿐이므로 이 안전계수는 보수적인 쪽으로만
    # 틀린다.
    print("\n  [보정용] Train 전용 헤드 적합...")
    calib_score = fit_score_heads(base, ("train",))
    calib_cost = fit_cost_heads(base, ("train",))
    print("  안전계수 선택 (정직한 헤드 기준)...")
    chosen = choose_safety(base, calib_score, calib_cost)

    splits = tuple(args.fit_splits)
    if splits == ("train",):
        score_heads, cost_heads = calib_score, calib_cost
    else:
        print(f"\n  [제출용] {'+'.join(splits)} 헤드 적합...")
        score_heads = fit_score_heads(base, splits)
        cost_heads = fit_cost_heads(base, splits)
    for model_id in MODELS:
        print(f"    {model_id:<12} smearing 계수 {cost_heads[model_id]['_smearing_factor']:.4f}")
    print(f"\n  {'등급':<10}{'안전계수':>10}{'보정셋 기대':>12}{'보정셋 초과확률':>16}")
    for tier in TIERS:
        safety, stats = chosen[tier]
        print(
            f"  {tier:<10}{safety:>10.4f}{stats['expected_score']:>12.4f}"
            f"{stats['overrun_rate']*100:>15.2f}%"
        )

    payload = _artifact_dict(
        base, score_heads, cost_heads, {t: chosen[t][0] for t in TIERS}
    )
    payload["training_summary"] = dict(payload["training_summary"])
    payload["training_summary"]["optimizer"] = (
        f"ridge-score-a{SCORE_ALPHA:g}+ridge-log-cost-a{COST_ALPHA:g}-smeared"
    )

    # 굽기 전에 공식 파서로 검증한다 — 스키마가 어긋나면 여기서 멈춘다.
    hash_regex.parse_artifact(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\n  검증 통과. 저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
