# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""학습 데이터를 늘리면 실제로 점수가 오르는가 — 정직하게 재는 법.

제출 아티팩트는 Train+Dev 2,640 문항으로 학습해도 된다(CHALLENGE_RULES.md). 그런데
그렇게 만든 헤드는 Dev 에 대해 in-sample 이라 우리 하네스로 검증할 수 없다.

우회로가 있다. Dev 를 반으로 갈라

    A안(현행)  Train 1,760 으로 헤드 적합
    B안        Train 1,760 + 보정 절반 440 = 2,200 으로 헤드 적합

둘 다 **같은 홀드아웃 440** 에서 채점하면 정직한 비교가 된다. 홀드아웃은 어느 쪽에도
학습에 쓰이지 않았다. 여기서 +440 의 효과가 측정되면 +880 의 방향은 같고 크기는
학습곡선의 감속을 감안해 그보다 작거나 비슷하다.

안전계수는 양쪽 모두 **각자의 헤드로 보정셋에서** 고른다 — 실제 배포와 같은 절차다.

## fork 제약

헤드 적합은 numpy 를 쓰므로 **부모에서만** 한다(macOS Accelerate 는 fork 후 교착).
자식에게는 이미 만들어진 예측 배열만 넘긴다.

## 사용

    .venv-train/bin/python analysis/more_data_check.py --splits 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.bake_artifact import (  # noqa: E402
    COST_ALPHA,
    SAFETY_RULE,
    SCORE_ALPHA,
    _artifact_dict,
)
from analysis.fast_score import outcome_table  # noqa: E402
from analysis.final_judgment import TIER_WEIGHT, assemble_predictions  # noqa: E402
from analysis.holdout_protocol import apply_policy, score_tier, subset  # noqa: E402
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

GRID_SIZE = 25
_CTX: Dict[str, Any] = {}


def fit_heads(
    score_x: Any, score_y: Any, cost_x: Any, cost_y: Any, rows: Sequence[int]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """지정한 행만 써서 점수·비용 헤드를 적합한다 (부모 전용, numpy)."""

    import numpy

    index = numpy.asarray(rows, dtype=int)
    sx, cx = score_x[index], cost_x[index]
    score_heads, cost_heads = {}, {}
    for model_id in MODELS:
        weights = fit_ridge(sx, score_y[model_id][index], l2=SCORE_ALPHA / len(index))
        score_heads[model_id] = {
            "intercept": float(weights[0]),
            "coefficients": [float(v) for v in weights[1:]],
        }
        target = numpy.log(numpy.maximum(cost_y[model_id][index], 1e-12))
        weights = fit_ridge(cx, target, l2=COST_ALPHA / len(index))
        predicted = numpy.exp(numpy.clip(cx @ weights, -50.0, 50.0))
        factor = float(
            numpy.mean(cost_y[model_id][index] / numpy.maximum(predicted, 1e-12))
        )
        cost_heads[model_id] = {
            "intercept": float(weights[0]) + math.log(factor),
            "coefficients": [float(v) for v in weights[1:]],
        }
    return score_heads, cost_heads


def evaluate(task: Tuple[int, str]) -> Tuple[int, str, float]:
    """한 분할 × 한 방식을 채점한다 (자식, 순수 파이썬)."""

    split_index, arm = task
    policy = _CTX["policy"]
    inputs, outcomes = _CTX["inputs"], _CTX["outcomes"]
    calib_index, holdout_index = _CTX["splits"][split_index]
    predictions = _CTX["predictions"][(split_index, arm)]

    calib = subset(inputs, outcomes, calib_index)
    holdout = subset(inputs, outcomes, holdout_index)
    calib_pred = [predictions[i] for i in calib_index]
    holdout_pred = [predictions[i] for i in holdout_index]
    calib_table = outcome_table(calib[0], calib[1], policy)

    weighted = Decimal(0)
    for tier in TIERS:
        safety, _stats = select_safety_risk(
            calib_pred, policy, tier, GRID_SIZE, calib_table, SAFETY_RULE
        )
        selected = apply_policy(holdout_pred, policy, tier, safety)
        row = score_tier(holdout[0], holdout[1], policy, tier, selected)
        raw = float(Decimal(row["tier_score"]))
        effective = raw if bool(row["budget_passed"]) else 0.0
        weighted += TIER_WEIGHT[tier] * Decimal(str(effective))
    return split_index, arm, float(weighted)


def main(argv: Sequence[str] | None = None) -> int:
    import random

    import numpy

    parser = argparse.ArgumentParser(prog="more-data-check")
    parser.add_argument("--splits", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    base = hash_regex.load_artifact(ROOT / "baselines" / "hash-regex-public.v1.json")
    policy = load_bundled_policy()
    inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    count = len(inputs.episodes)

    print("=" * 74)
    print("학습 데이터 증가 효과 — 정직한 홀드아웃")
    print(f"  A안  Train 1,760")
    print(f"  B안  Train 1,760 + 보정 절반 440 = 2,200")
    print(f"  분할 {args.splits}회, 양쪽 모두 같은 홀드아웃 440 에서 채점")
    print("=" * 74)

    print("\n  학습 행렬 준비...")
    score_x, score_y, _t = build_multi(("train", "dev"), base)
    cost_rows, cost_cols = [], None
    for split in ("train", "dev"):
        matrix, cost, _ids = build_matrix(split, base)
        cost_rows.append(matrix)
        cost_cols = (
            {m: [cost[m]] for m in cost}
            if cost_cols is None
            else {m: cost_cols[m] + [cost[m]] for m in cost}
        )
    cost_x = numpy.vstack(cost_rows)
    cost_y = {m: numpy.concatenate(v) for m, v in cost_cols.items()}
    train_rows = list(range(score_x.shape[0] - count))  # Dev 앞이 Train
    dev_offset = len(train_rows)

    splits = []
    for trial in range(args.splits):
        order = list(range(count))
        random.Random(args.seed + trial).shuffle(order)
        splits.append((order[: count // 2], order[count // 2 :]))

    print(f"  헤드 적합 {args.splits * 2}세트 (부모에서, numpy)...")
    predictions: Dict[Tuple[int, str], Any] = {}
    shared = fit_heads(score_x, score_y, cost_x, cost_y, train_rows)
    for index, (calib_index, _holdout) in enumerate(splits):
        for arm in ("train_only", "plus_calib"):
            if arm == "train_only":
                heads = shared
            else:
                rows = train_rows + [dev_offset + i for i in calib_index]
                heads = fit_heads(score_x, score_y, cost_x, cost_y, rows)
            staged = hash_regex.parse_artifact(
                _artifact_dict(base, heads[0], heads[1], base.tier_safety_ratios)
            )
            predictions[(index, arm)] = assemble_predictions(
                inputs.episodes, staged, None
            )
        if (index + 1) % 10 == 0:
            print(f"    {index + 1}/{args.splits}", flush=True)

    _CTX.update(
        policy=policy, inputs=inputs, outcomes=outcomes,
        splits=splits, predictions=predictions,
    )

    tasks = [(i, arm) for i in range(len(splits)) for arm in ("train_only", "plus_calib")]
    results: Dict[str, List[float]] = {"train_only": [0.0] * len(splits),
                                       "plus_calib": [0.0] * len(splits)}
    jobs = max(1, args.jobs)
    print(f"\n  채점 {len(tasks)}회 (병렬 {jobs})...")
    if jobs > 1:
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for done, (index, arm, value) in enumerate(
                pool.imap_unordered(evaluate, tasks, chunksize=1), start=1
            ):
                results[arm][index] = value
                if done % 20 == 0 or done == len(tasks):
                    print(f"    {done}/{len(tasks)}", flush=True)
    else:
        for task in tasks:
            index, arm, value = evaluate(task)
            results[arm][index] = value

    a, b = results["train_only"], results["plus_calib"]
    diff = [y - x for x, y in zip(a, b)]
    mean = statistics.fmean(diff)
    stderr = statistics.stdev(diff) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0

    print("\n결과")
    print(f"  A안 (Train 1,760)          {statistics.fmean(a):.4f}")
    print(f"  B안 (Train+보정 2,200)      {statistics.fmean(b):.4f}")
    print(f"  차이                       {mean:+.4f}  (쌍대 SE {stderr:.4f}, "
          f"t={mean / stderr if stderr else float('nan'):.2f})")
    print(f"  B안이 이긴 분할            {sum(1 for d in diff if d > 0)}/{len(diff)}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "report_type": "more-data-check-v1",
                    "splits": len(splits),
                    "train_only": a,
                    "plus_calib": b,
                    "mean_difference": mean,
                    "paired_stderr": stderr,
                },
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
