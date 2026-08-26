# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""비용을 분위수 회귀로 예측한다 — 되돌림 편향 없이.

## 왜

현재 baseline은 로그 비용을 최소제곱으로 회귀하고 exp를 씌운다. 최소제곱은 조건부
평균을 추정하므로 exp(E[log y]) 를 얻는데, 이는 E[y] 가 아니라 기하평균에 가깝다.
잔차항 E[exp(u)] 가 빠지면서 총합이 체계적으로 낮아진다 (light 18.2% / think 52.2%).

Duan smearing 은 이를 사후 보정하지만 전역 계수 하나여서 이분산·꼬리에 약하고,
집단별 계수로 나눠도 듣지 않았다 (decision-trail.md §8, §9 — 꼬리 지배).

**분위수는 단조변환과 교환된다.** 로그 공간에서 tau 분위수를 추정하고 exp를 씌우면
원래 단위의 정확한 tau 분위수다. 보정이 필요 없다. 그리고 예산 판정에 필요한 것은
평균이 아니라 상한이므로, 상위 분위수가 목적에 그대로 맞는다.

## 방법

pinball(check) 손실의 경사하강으로 선형 분위수 회귀를 적합한다.

    loss(r) = max(tau*r, (tau-1)*r),   r = y - Xw

닫힌 형태가 없어 반복 최적화가 필요하지만 학습은 오프라인이다. 추론은 내적 하나이므로
제출 컨테이너의 표준 라이브러리 제약과 무관하다.

특징은 baseline artifact 와 동일하게 쓴다(표준화 포함). 특징을 바꾸면 원인이 섞이므로
여기서는 **추정 방식만** 교체해 비교한다.

## 사용

    .venv-train/bin/python analysis/quantile_cost.py --report analysis/reports/quantile.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402

MODELS = ("ax31-light", "ax31", "axk1-think")
LABEL = {"ax31-light": "싼 모델", "ax31": "중간 모델", "axk1-think": "비싼 추론 모델"}
RATES = {
    "ax31-light": (Decimal("1"), Decimal("4")),
    "ax31": (Decimal("2.127"), Decimal("8.509")),
    "axk1-think": (Decimal("6.565"), Decimal("26.260")),
}
TOKEN_UNIT = Decimal("1000000")


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "numpy 필요 (학습 전용). .venv-train/bin/python 으로 실행하십시오."
        )
    return numpy


def actual_cost(model_id: str, entry: Mapping[str, Any]) -> float:
    rate_in, rate_out = RATES[model_id]
    return float(
        (
            Decimal(entry["input_tokens"]) * rate_in
            + Decimal(entry["output_tokens"]) * rate_out
        )
        / TOKEN_UNIT
    )


def build_matrix(split: str, artifact: Any) -> Tuple[Any, Mapping[str, Any], List[str]]:
    """baseline 과 동일한 표준화 특징 행렬 + 모델별 실제 비용."""

    numpy = _require_numpy()
    episodes = load_input(
        ROOT / "data" / "materialized" / split / "inputs.json"
    ).episodes
    outcomes = {
        row["episode_id"]: row["models"]
        for row in json.loads(
            (ROOT / "data" / split / "outcomes.json").read_text("utf-8")
        )["episodes"]
    }
    mean = numpy.asarray(artifact.feature_mean, dtype=numpy.float64)
    scale = numpy.asarray(artifact.feature_scale, dtype=numpy.float64)

    rows = []
    costs: Dict[str, List[float]] = {m: [] for m in MODELS}
    ids: List[str] = []
    for episode in episodes:
        raw = numpy.asarray(
            hash_regex.raw_feature_vector(episode, artifact.hash_bins),
            dtype=numpy.float64,
        )
        rows.append((raw - mean) / scale)
        for model_id in MODELS:
            costs[model_id].append(
                actual_cost(model_id, outcomes[episode.episode_id][model_id])
            )
        ids.append(episode.episode_id)

    matrix = numpy.asarray(rows, dtype=numpy.float64)
    bias = numpy.ones((matrix.shape[0], 1), dtype=numpy.float64)
    return (
        numpy.hstack([bias, matrix]),
        {m: numpy.asarray(v, dtype=numpy.float64) for m, v in costs.items()},
        ids,
    )


def fit_quantile(
    matrix: Any,
    target: Any,
    tau: float,
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
    seed: int,
) -> Any:
    """pinball 손실 + L2 벌점을 경사하강으로 적합한다.

    pinball 손실의 기울기는 잔차 부호에만 의존한다 (tau 또는 tau-1). 부드럽지 않지만
    구간별 선형이라 안정적이다. 학습률을 코사인으로 감쇠시켜 수렴시킨다.
    절편(0번 열)에는 벌점을 주지 않는다.
    """

    numpy = _require_numpy()
    rows, columns = matrix.shape
    weights = numpy.zeros(columns, dtype=numpy.float64)
    weights[0] = float(numpy.quantile(target, tau))  # 절편을 무조건부 분위수로 시작
    penalty_mask = numpy.ones(columns, dtype=numpy.float64)
    penalty_mask[0] = 0.0

    for step in range(iterations):
        residual = target - matrix @ weights
        # d/dw pinball = -(tau - 1{r<0}) * x
        indicator = numpy.where(residual < 0.0, tau - 1.0, tau)
        gradient = -(matrix.T @ indicator) / rows + l2 * penalty_mask * weights
        rate = learning_rate * 0.5 * (1.0 + math.cos(math.pi * step / iterations))
        weights -= rate * gradient
    return weights


def fit_least_squares(matrix: Any, target: Any, *, l2: float) -> Any:
    """비교용 — baseline 과 같은 ridge (닫힌 형태).

    손실을 행 수로 나눈 규약을 쓴다. 분위수 회귀의 경사도 같은 규약이므로
    두 방식이 같은 l2 값을 같은 의미로 쓴다. baseline 학습기의 alpha=100 에
    대응하는 값은 100/행수 이다.
    """

    numpy = _require_numpy()
    rows, columns = matrix.shape
    penalty = numpy.eye(columns) * l2
    penalty[0, 0] = 0.0
    return numpy.linalg.solve(
        matrix.T @ matrix / rows + penalty, matrix.T @ target / rows
    )


def pinball_loss(residual: Any, tau: float) -> float:
    numpy = _require_numpy()
    return float(numpy.mean(numpy.maximum(tau * residual, (tau - 1.0) * residual)))


def evaluate(
    train: Tuple[Any, Mapping[str, Any], List[str]],
    dev: Tuple[Any, Mapping[str, Any], List[str]],
    taus: Sequence[float],
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
    seed: int,
) -> Mapping[str, Any]:
    """모델별로 ridge(로그) + smearing 과 분위수 회귀를 비교한다."""

    numpy = _require_numpy()
    xtr, ytr, _ = train
    xdv, ydv, _ = dev
    out: Dict[str, Any] = {"taus": list(taus), "models": {}}

    for model_id in MODELS:
        log_tr = numpy.log(numpy.maximum(ytr[model_id], 1e-12))
        actual_dev = ydv[model_id]
        actual_sum = float(actual_dev.sum())

        entry: Dict[str, Any] = {"actual_sum_dev": actual_sum, "methods": {}}

        # (a) 현재 방식 재현 — ridge on log, exp
        w_ls = fit_least_squares(xtr, log_tr, l2=l2)
        pred_dev = numpy.exp(numpy.clip(xdv @ w_ls, -50.0, 50.0))
        entry["methods"]["ridge_log_exp"] = {
            "sum": float(pred_dev.sum()),
            "bias": actual_sum / float(pred_dev.sum()) - 1.0,
            "exceed_rate": float(numpy.mean(actual_dev > pred_dev)),
        }

        # (b) + Duan smearing (Train 잔차로 계수 산출)
        pred_tr = numpy.exp(numpy.clip(xtr @ w_ls, -50.0, 50.0))
        factor = float(numpy.mean(ytr[model_id] / numpy.maximum(pred_tr, 1e-12)))
        smeared = pred_dev * factor
        entry["methods"]["ridge_log_exp_smeared"] = {
            "smearing_factor": factor,
            "sum": float(smeared.sum()),
            "bias": actual_sum / float(smeared.sum()) - 1.0,
            "exceed_rate": float(numpy.mean(actual_dev > smeared)),
        }

        # (c) 분위수 회귀 — 로그 공간에서 적합 후 exp
        for tau in taus:
            w_q = fit_quantile(
                xtr,
                log_tr,
                tau,
                l2=l2,
                iterations=iterations,
                learning_rate=learning_rate,
                seed=seed,
            )
            pred_q = numpy.exp(numpy.clip(xdv @ w_q, -50.0, 50.0))
            pred_q_train = numpy.exp(numpy.clip(xtr @ w_q, -50.0, 50.0))
            entry["methods"][f"quantile_{tau:g}"] = {
                "sum": float(pred_q.sum()),
                "bias": actual_sum / float(pred_q.sum()) - 1.0,
                # 분위수의 직접 검증: 실제가 예측을 넘는 비율이 1-tau 에 가까워야 한다.
                # train 이 목표에서 벗어나면 미수렴, train 만 맞으면 일반화 실패.
                "exceed_rate": float(numpy.mean(actual_dev > pred_q)),
                "exceed_rate_train": float(
                    numpy.mean(ytr[model_id] > pred_q_train)
                ),
                "exceed_rate_target": 1.0 - tau,
                "pinball_dev": pinball_loss(
                    numpy.log(numpy.maximum(actual_dev, 1e-12)) - (xdv @ w_q), tau
                ),
            }
        out["models"][model_id] = entry
    return out


def report(res: Mapping[str, Any]) -> None:
    print("\n[1] 총합 편향 — 실제합/예측합 − 1  (0에 가까울수록 무편향)")
    methods = ["ridge_log_exp", "ridge_log_exp_smeared"] + [
        f"quantile_{t:g}" for t in res["taus"]
    ]
    width = max(len(m) for m in methods) + 2
    print(f"  {'방식':<{width}}" + "".join(f"{LABEL[m]:>14}" for m in MODELS))
    for method in methods:
        cells = ""
        for model_id in MODELS:
            b = res["models"][model_id]["methods"][method]["bias"]
            cells += f"{b*100:>13.1f}%"
        print(f"  {method:<{width}}{cells}")

    print("\n[2] 분위수 교정 — 실제가 예측을 넘는 비율 (목표 = 1−tau), Dev")
    print(f"  {'방식':<{width}}{'목표':>8}" + "".join(f"{LABEL[m]:>14}" for m in MODELS))
    for method in methods:
        first = res["models"][MODELS[0]]["methods"][method]
        target = first.get("exceed_rate_target")
        cells = ""
        for model_id in MODELS:
            r = res["models"][model_id]["methods"][method]["exceed_rate"]
            cells += f"{r*100:>13.1f}%"
        tgt = f"{target*100:>7.0f}%" if target is not None else f"{'—':>8}"
        print(f"  {method:<{width}}{tgt}{cells}")

    print("\n[3] 같은 지표를 Train 에서 — 목표에서 벗어나면 미수렴")
    print(f"  {'방식':<{width}}{'목표':>8}" + "".join(f"{LABEL[m]:>14}" for m in MODELS))
    for method in methods:
        first = res["models"][MODELS[0]]["methods"][method]
        if "exceed_rate_train" not in first:
            continue
        target = first["exceed_rate_target"]
        cells = "".join(
            f"{res['models'][m]['methods'][method]['exceed_rate_train']*100:>13.1f}%"
            for m in MODELS
        )
        print(f"  {method:<{width}}{target*100:>7.0f}%{cells}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantile-cost",
        description="분위수 회귀로 비용을 예측하고 현재 방식과 비교합니다.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "baselines" / "hash-regex-public.v1.json",
    )
    parser.add_argument(
        "--tau", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95]
    )
    # baseline 학습기의 ridge_alpha=100 을 Train 1760행 규약으로 환산한 값
    parser.add_argument("--l2", type=float, default=100.0 / 1760.0)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = hash_regex.load_artifact(args.artifact)

    print("=" * 74)
    print("분위수 회귀 vs 현재 방식 — Train 1760 적합, Dev 880 검증")
    print(f"  tau {args.tau} · L2 {args.l2} · 반복 {args.iterations}")
    print("=" * 74)

    train = build_matrix("train", artifact)
    dev = build_matrix("dev", artifact)
    print(f"  특징 차원 {train[0].shape[1]} (절편 포함)")

    res = evaluate(
        train,
        dev,
        args.tau,
        l2=args.l2,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    report(res)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.report}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
