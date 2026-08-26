# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""최종 판정 — 비용 추정 방식을 바꿨을 때 실제로 점수가 오르는가.

예측기 단독 지표(편향·교정)로는 답이 안 나온다. 예측한 비용을 실제 배분기에 넣고
공식 채점기로 채점해야 한다. 그리고 보정셋에서 안전계수를 고르고 홀드아웃에서
측정해야 한다 — 그러지 않으면 baseline 이 저지른 in-sample 보고를 반복한다.

## 판정 기준 (사전 고정)

    예산 초과 확률은 같거나 낮으면서 기대점수는 높을 것.
    둘 중 하나만 만족하면 실패로 간주한다.

기대점수는 초과 시 0을 적용한 등급점수의 평균이다. 예산을 넘기면 그 등급은 0점이므로,
초과를 무시한 평균은 의미가 없다.

## 비교 대상

점수 예측(score_heads)과 특징은 공개 artifact 를 그대로 쓴다. **비용 추정만** 교체해
원인을 섞지 않는다. 배분기(라그랑주 + 이분탐색)도 그대로 둔다.

    shipped              공개 artifact 의 log_cost_heads (현재 방식)
    ridge_log_exp        같은 방식을 직접 재적합 (재현 확인용)
    ridge_smeared        + Duan smearing 전역 보정
    quantile_0.8/0.9/0.95  로그 공간 분위수 회귀 후 exp

## 사용

    .venv-train/bin/python analysis/final_judgment.py --report analysis/reports/final.json
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.holdout_protocol import (  # noqa: E402
    _safety_grid,
    apply_policy,
    build_splits,
    score_tier,
    select_safety,
    subset,
    train_vocabulary,
)
from analysis.fast_score import outcome_table  # noqa: E402
from analysis.risk_safety import select_safety_risk  # noqa: E402
from analysis.quantile_cost import build_matrix  # noqa: E402
from analysis.quantile_cost import (  # noqa: E402
    MODELS,
    build_matrix,
    fit_least_squares,
    fit_quantile,
)
from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

TIER_WEIGHT = {"fast": Decimal("0.4"), "balanced": Decimal("0.3"), "premium": Decimal("0.3")}


def _numpy() -> Any:
    import numpy

    return numpy


def build_cost_estimators(
    artifact: Any, taus: Sequence[float], *, l2: float, iterations: int,
    learning_rate: float, seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    """Train 에서 비용 추정기들을 적합해 Dev 예측을 만든다.

    반환: 방식 이름 -> {model_id: Dev 문항별 예측 비용 배열}
    """

    numpy = _numpy()
    xtr, ytr, _ = build_matrix("train", artifact)
    xdv, _ydv, _ = build_matrix("dev", artifact)
    out: Dict[str, Dict[str, Any]] = {}

    for model_id in MODELS:
        log_target = numpy.log(numpy.maximum(ytr[model_id], 1e-12))

        weights = fit_least_squares(xtr, log_target, l2=l2)
        pred_dev = numpy.exp(numpy.clip(xdv @ weights, -50.0, 50.0))
        out.setdefault("ridge_log_exp", {})[model_id] = pred_dev

        pred_train = numpy.exp(numpy.clip(xtr @ weights, -50.0, 50.0))
        factor = float(
            numpy.mean(ytr[model_id] / numpy.maximum(pred_train, 1e-12))
        )
        out.setdefault("ridge_smeared", {})[model_id] = pred_dev * factor

        for tau in taus:
            w = fit_quantile(
                xtr, log_target, tau, l2=l2, iterations=iterations,
                learning_rate=learning_rate, seed=seed,
            )
            out.setdefault(f"quantile_{tau:g}", {})[model_id] = numpy.exp(
                numpy.clip(xdv @ w, -50.0, 50.0)
            )
    return out


def assemble_predictions(
    episodes: Sequence[Any],
    artifact: Any,
    costs: Mapping[str, Any] | None,
    truth: Mapping[str, Any] | None = None,
    scores_override: Mapping[str, Any] | None = None,
) -> List[Tuple[Mapping[str, float], Mapping[str, float]]]:
    """(점수 예측, 비용 예측) 쌍을 배분기가 먹는 형태로 만든다.

    costs 가 None 이면 비용도 공개 artifact 값 (= shipped 방식).

    truth 는 진단 전용 반칙 입력이다. {"quality": ..., "cost": ...} 중 있는 것을
    참값으로 갈아끼운다 — "그 예측이 완벽했다면 얼마나 오르는가"의 상한을 재기 위한
    것이고, 제출 경로에는 쓰이지 않는다.
    """

    predictions = []
    for index, episode in enumerate(episodes):
        scores, shipped_costs = hash_regex.predict_episode(episode, artifact)
        if scores_override is not None:
            scores = {m: float(scores_override[m][index]) for m in MODELS}
        if costs is not None:
            shipped_costs = {m: float(costs[m][index]) for m in MODELS}
        if truth is not None:
            if "quality" in truth:
                scores = {
                    m: (
                        truth["quality"][m][index]
                        if m in truth["quality"] else scores[m]
                    )
                    for m in MODELS
                }
            if "cost" in truth:
                shipped_costs = {m: truth["cost"][m][index] for m in MODELS}
        predictions.append((scores, shipped_costs))
    return predictions


_CTX: Dict[str, Any] = {}


def flatten_splits(
    plans: Mapping[str, Sequence[Tuple[List[int], List[int]]]]
) -> List[Tuple[str, List[int], List[int]]]:
    """분할을 순서가 고정된 리스트로 편다.

    방식 간 쌍대 비교가 성립하려면 모든 방식이 **같은 순서의 같은 분할**을 봐야 한다.
    """

    out: List[Tuple[str, List[int], List[int]]] = []
    for name, split_plans in plans.items():
        for calib_index, holdout_index in split_plans:
            out.append((name, calib_index, holdout_index))
    return out


def evaluate_split(task: Tuple[int, int]) -> Tuple[int, int, Dict[str, Any]]:
    """(방식, 분할) 하나를 보정→홀드아웃으로 채점한다.

    입력·예측은 인자가 아니라 부모가 채운 _CTX 에서 읽는다. 워커마다 예측 배열을
    피클링하면 계산보다 전송이 비싸지기 때문이다 (fork 로 물려받는다).

    safety_policy
        maxscore  baseline 규칙 — 보정셋 점수를 최대화하는 안전계수를 고른다.
                  비용 추정이 보수적이면 이 규칙이 안전계수를 끌어올려 보수성을 상쇄한다.
        trust     안전계수 1.0 고정 — 추정기의 보수성을 그대로 신뢰한다.
                  보정셋 탐색이 없으므로 in-sample 과적합 경로 자체가 사라진다.
    """

    method_index, split_index = task
    inputs = _CTX["inputs"]
    outcomes = _CTX["outcomes"]
    policy = _CTX["policy"]
    grid_size = _CTX["grid_size"]
    predictions, safety_policy = _CTX["methods"][method_index]
    per_tier_predictions = _CTX.get("per_tier", {}).get(method_index)
    name, calib_index, holdout_index = _CTX["splits"][split_index]

    calib = subset(inputs, outcomes, calib_index)
    holdout = subset(inputs, outcomes, holdout_index)
    calib_pred = [predictions[i] for i in calib_index]
    holdout_pred = [predictions[i] for i in holdout_index]

    # 위험 인지 규칙은 보정셋의 문항별 비용·품질 표가 필요하다 (닫힌 형태 추정용).
    calib_table = None
    if safety_policy not in ("trust", "maxscore", "all_light"):
        calib_table = outcome_table(calib[0], calib[1], policy)

    rows: Dict[str, Any] = {}
    base_calib_pred, base_holdout_pred = calib_pred, holdout_pred
    for tier in TIERS:
        if per_tier_predictions is not None:
            # 등급별 점수 헤드 — 등급마다 예산 여유가 달라(1.25 / 2.0 / 4.0) 필요한
            # 판별이 다르다. fast 는 light↔ax31, premium 은 think 쪽이 관건이다.
            tier_pred = per_tier_predictions[tier]
            calib_pred = [tier_pred[i] for i in calib_index]
            holdout_pred = [tier_pred[i] for i in holdout_index]
        else:
            calib_pred, holdout_pred = base_calib_pred, base_holdout_pred
        if safety_policy == "all_light":
            # 위험 0 기준선 — 배분기를 거치지 않는다.
            safety = 0.0
            selected = tuple(
                policy.light_model_id for _ in holdout[0].episodes
            )
        elif safety_policy == "trust":
            safety = 1.0
            selected = apply_policy(holdout_pred, policy, tier, safety)
        elif safety_policy.startswith("gridfrac"):
            # 고정 안전계수 — 보정셋 탐색 없이 그리드의 지정 지점을 그대로 쓴다.
            # 실제 제출이 하는 일이고, 적응적 선택이 이보다 나은지 재는 기준이기도 하다.
            grid = _safety_grid(policy, tier, grid_size)
            spec = safety_policy[len("gridfrac"):]
            if ":" in spec:
                # 등급별 위치 — fast:balanced:premium. 등급마다 예산 여유가 달라
                # (배수 1.25 / 2.0 / 4.0) 하나의 값으로 셋을 다 맞출 수 없다.
                order = {"fast": 0, "balanced": 1, "premium": 2}
                fraction = float(spec.split(":")[order[tier]])
            else:
                fraction = float(spec)
            safety = grid[min(len(grid) - 1, int(round(fraction * (len(grid) - 1))))]
            selected = apply_policy(holdout_pred, policy, tier, safety)
        elif safety_policy == "oracle":
            # 진단 전용 — 홀드아웃을 보고 고른다(반칙). 어떤 선택 규칙도 이 값을
            # 넘을 수 없으므로, 우리 규칙과의 격차가 곧 "선택에 남은 여지"다.
            best_row = None
            for candidate in _safety_grid(policy, tier, grid_size):
                trial = apply_policy(holdout_pred, policy, tier, candidate)
                probe = score_tier(holdout[0], holdout[1], policy, tier, trial)
                value = (
                    float(Decimal(probe["tier_score"]))
                    if bool(probe["budget_passed"]) else 0.0
                )
                if best_row is None or value > best_row[0]:
                    best_row = (value, candidate, trial)
            safety, selected = best_row[1], best_row[2]
        elif safety_policy == "maxscore":
            safety, _calib_report = select_safety(
                calib[0], calib[1], policy, calib_pred, tier, grid_size
            )
            selected = apply_policy(holdout_pred, policy, tier, safety)
        else:
            safety, _calib_report = select_safety_risk(
                calib_pred, policy, tier, grid_size, calib_table, safety_policy,
            )
            selected = apply_policy(holdout_pred, policy, tier, safety)
        report_row = score_tier(holdout[0], holdout[1], policy, tier, selected)
        passed = bool(report_row["budget_passed"])
        raw = float(Decimal(report_row["tier_score"]))
        rows[tier] = {
            "split": name,
            "safety_ratio": safety,
            "budget_ratio": float(Decimal(report_row["budget_ratio"])),
            "budget_passed": passed,
            # 초과 시 0 — 실제 채점 규칙
            "effective_score": raw if passed else 0.0,
            "raw_score": raw,
        }
    return method_index, split_index, rows


def summarize(
    method: str,
    splits: Sequence[Tuple[str, List[int], List[int]]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """분할별 결과를 집계한다.

    무작위 분할과 층화 분할을 **섞어서 표준오차를 내지 않는다.** 무작위 분할은 같은
    분포에서 반복 추출한 표본이라 평균의 오차가 1/√n 으로 줄지만, 층화 8방향은
    결정적인 스트레스 시험이라 반복해도 값이 변하지 않는다. 둘을 합쳐 stdev/√n 을
    내면 층화 실패가 '표본 잡음'으로 희석돼 보인다.
    """

    weighted: List[float] = []
    for row in rows:
        total = Decimal(0)
        for tier in TIERS:
            total += TIER_WEIGHT[tier] * Decimal(str(row[tier]["effective_score"]))
        weighted.append(float(total))

    names = [s[0] for s in splits]
    random_scores = [w for w, n in zip(weighted, names) if n == "random"]
    strat_scores = {n: w for n, w in zip(names, weighted) if n != "random"}

    summary: Dict[str, Any] = {
        "method": method,
        "split_names": names,
        "per_split_weighted": weighted,
        # 분할별 등급 점수 — 등급 조합의 분포를 재실행 없이 합성하기 위해 남긴다.
        "per_split_tier": {
            tier: [row[tier]["effective_score"] for row in rows] for tier in TIERS
        },
        "random_trials": len(random_scores),
        "stratified_weighted": strat_scores,
        "tiers": {},
    }
    if random_scores:
        summary["random_mean"] = statistics.fmean(random_scores)
    if len(random_scores) > 1:
        stdev = statistics.stdev(random_scores)
        summary["random_stdev"] = stdev
        # 무작위 분할에 한해 유효한 평균의 표준오차
        summary["random_stderr"] = stdev / math.sqrt(len(random_scores))
    if strat_scores:
        summary["stratified_mean"] = statistics.fmean(strat_scores.values())

    for tier in TIERS:
        trials = [row[tier] for row in rows]
        strat = [r for r in trials if r["split"] != "random"]
        summary["tiers"][tier] = {
            "trials": len(trials),
            "pass_rate": statistics.fmean(r["budget_passed"] for r in trials),
            "expected_score": statistics.fmean(r["effective_score"] for r in trials),
            "raw_score_when_passed": (
                statistics.fmean(r["raw_score"] for r in trials if r["budget_passed"])
                if any(r["budget_passed"] for r in trials)
                else 0.0
            ),
            "worst_budget_ratio": max(r["budget_ratio"] for r in trials),
            "median_safety_ratio": statistics.median(
                r["safety_ratio"] for r in trials
            ),
            "stratified_pass_rate": (
                statistics.fmean(r["budget_passed"] for r in strat) if strat else None
            ),
        }
    summary["expected_final_score"] = float(
        sum(
            TIER_WEIGHT[tier] * Decimal(str(summary["tiers"][tier]["expected_score"]))
            for tier in TIERS
        )
    )
    return summary


def run_judgments(
    method_names: Sequence[str],
    splits: Sequence[Tuple[str, List[int], List[int]]],
    *,
    jobs: int,
) -> List[Dict[str, Any]]:
    """_CTX 에 실린 모든 (방식, 분할) 조합을 채점하고 방식별로 집계한다."""

    tasks = [
        (m, s) for m in range(len(method_names)) for s in range(len(splits))
    ]
    collected: List[List[Any]] = [
        [None] * len(splits) for _ in method_names
    ]

    if jobs > 1:
        import multiprocessing

        # fork 로 _CTX 를 물려준다 — spawn 이면 자식이 부모 전역을 못 본다.
        # 분할 채점은 순수 파이썬이고 numpy 적합은 이 시점에 이미 끝나 있다.
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=jobs) as pool:
            done = 0
            for m, s, rows in pool.imap_unordered(
                evaluate_split, tasks, chunksize=1
            ):
                collected[m][s] = rows
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"    {done}/{len(tasks)}", flush=True)
    else:
        for index, task in enumerate(tasks, start=1):
            m, s, rows = evaluate_split(task)
            collected[m][s] = rows
            if index % 50 == 0 or index == len(tasks):
                print(f"    {index}/{len(tasks)}", flush=True)

    return [
        summarize(name, splits, collected[m])
        for m, name in enumerate(method_names)
    ]


def paired_comparison(
    results: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """기준 방식 대비 분할별 차이를 짝지어 비교한다.

    방식들은 같은 분할을 보므로 '이 분할이 어려웠다'는 성분이 차이에서 상쇄된다.
    주변 평균끼리 비교하면 그 공통 성분이 두 번 들어가 오차가 부풀려진다.
    """

    usable = [r for r in results if r.get("random_trials", 0) > 1]
    if not usable:
        return []
    reference = max(usable, key=lambda r: r["random_mean"])
    names = reference["split_names"]
    picks = [i for i, n in enumerate(names) if n == "random"]
    base = [reference["per_split_weighted"][i] for i in picks]

    out: List[Dict[str, Any]] = []
    for r in usable:
        if r["method"] == reference["method"]:
            continue
        other = [r["per_split_weighted"][i] for i in picks]
        diff = [b - o for b, o in zip(base, other)]
        mean = statistics.fmean(diff)
        stderr = statistics.stdev(diff) / math.sqrt(len(diff))
        unpaired = math.sqrt(
            reference.get("random_stderr", 0.0) ** 2
            + r.get("random_stderr", 0.0) ** 2
        )
        out.append(
            {
                "reference": reference["method"],
                "method": r["method"],
                "mean_difference": mean,
                "paired_stderr": stderr,
                "t_statistic": mean / stderr if stderr else float("nan"),
                "unpaired_stderr": unpaired,
                "significant_95": bool(stderr and abs(mean) > 1.96 * stderr),
            }
        )
    return out


def report(results: Sequence[Mapping[str, Any]]) -> None:
    width = max(len(r["method"]) for r in results) + 2

    print("\n[1] 홀드아웃 예산 통과율 — 높을수록 안전")
    print(
        f"  {'방식':<{width}}{'fast':>9}{'balanced':>10}{'premium':>9}"
        f"{'층화만':>9}"
    )
    for r in results:
        cells = "".join(
            f"{r['tiers'][t]['pass_rate']*100:>8.0f}%" for t in TIERS
        )
        strat = statistics.fmean(
            r["tiers"][t]["stratified_pass_rate"] for t in TIERS
        )
        print(f"  {r['method']:<{width}}{cells}{strat*100:>8.0f}%")

    print("\n[2] 기대점수 — 초과 시 0 적용 (실제 채점 규칙)")
    print(
        f"  {'방식':<{width}}{'fast':>9}{'balanced':>10}{'premium':>9}"
        f"{'가중 최종':>11}"
    )
    for r in results:
        cells = "".join(
            f"{r['tiers'][t]['expected_score']:>9.4f}" for t in TIERS
        )
        print(f"  {r['method']:<{width}}{cells}{r['expected_final_score']:>11.4f}")

    print("\n[2b] 무작위 분할과 층화 분할을 분리 — 섞어서 평균내면 층화 실패가 희석된다")
    print(
        f"  {'방식':<{width}}{'무작위평균':>11}{'표준오차':>10}"
        f"{'층화평균':>10}{'무작위n':>9}"
    )
    for r in results:
        stderr = r.get("random_stderr")
        cell = f"{stderr:>10.4f}" if stderr is not None else f"{'-':>10}"
        print(
            f"  {r['method']:<{width}}{r.get('random_mean', float('nan')):>11.4f}"
            f"{cell}{r.get('stratified_mean', float('nan')):>10.4f}"
            f"{r.get('random_trials', 0):>9}"
        )

    pairs = paired_comparison(results)
    if pairs:
        reference = pairs[0]["reference"]
        print(
            f"\n[4] 쌍대 비교 — 같은 분할끼리 짝지어 차이를 잰다 (기준: {reference})"
        )
        print("     같은 분할을 공유하므로 '분할 난이도' 성분이 차이에서 상쇄된다.")
        print(
            f"  {'방식':<{width}}{'평균차':>9}{'쌍대SE':>9}{'t':>7}"
            f"{'비쌍대SE':>10}{'유의':>6}"
        )
        for row in pairs:
            mark = "예" if row["significant_95"] else "아니오"
            print(
                f"  {row['method']:<{width}}{row['mean_difference']:>9.4f}"
                f"{row['paired_stderr']:>9.4f}{row['t_statistic']:>7.2f}"
                f"{row['unpaired_stderr']:>10.4f}{mark:>6}"
            )
        best = min(row["paired_stderr"] for row in pairs)
        print(
            f"\n  판정 해상도: 쌍대 표준오차 최소 {best:.4f}"
            f" → 95%로 구분 가능한 최소 격차 {1.96 * best:.4f}"
        )

    print("\n[3] 통과했을 때의 점수 / 최악 비용비율 / 선택된 안전계수 중앙값")
    for tier in TIERS:
        print(f"  ── {tier}")
        print(
            f"     {'방식':<{width}}{'통과시점수':>11}{'최악비율':>10}{'안전계수':>10}"
        )
        for r in results:
            t = r["tiers"][tier]
            print(
                f"     {r['method']:<{width}}{t['raw_score_when_passed']:>11.4f}"
                f"{t['worst_budget_ratio']:>10.3f}{t['median_safety_ratio']:>10.4f}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="final-judgment",
        description="비용 추정 방식을 배분기에 연결해 홀드아웃에서 최종 판정합니다.",
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
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="분할 채점 병렬도. 1이면 단일 프로세스 (결과는 동일)",
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--risk-rules", nargs="*", default=[],
        help="위험 인지 안전계수 규칙: expected | cp5 | cp10 | cp20 ...",
    )
    parser.add_argument(
        "--risk-bases", nargs="*", default=["shipped"],
        help="위 규칙을 적용할 비용 추정 방식",
    )
    parser.add_argument(
        "--tier-heads", nargs="*", default=[],
        help="등급별 점수 헤드 'fast/balanced/premium' (--score-heads 이름 사용)",
    )
    parser.add_argument(
        "--cost-alphas", type=float, nargs="*", default=[],
        help="비용 헤드 ridge alpha 변형 (smearing 포함)",
    )
    parser.add_argument("--calib-resamples", type=int, default=500)
    parser.add_argument(
        "--truth-models", nargs="*", default=[],
        help="--truth score 를 일부 모델에만 적용 (병목 분리용)",
    )
    parser.add_argument(
        "--score-heads", nargs="*", default=[],
        help="점수 헤드 재적합: ridge<alpha> | binomial<alpha> (예: binomial1000)",
    )
    parser.add_argument(
        "--fixed-tier-fracs", nargs="*", default=[],
        help="등급별 고정 안전계수 위치 'fast:balanced:premium' (예: 0.2:0.6:0.6)",
    )
    parser.add_argument(
        "--truth", nargs="*", default=[], choices=["score", "cost"],
        help="진단 전용 — 해당 예측을 참값으로 대체해 상한을 잰다",
    )
    parser.add_argument(
        "--fixed-fracs", type=float, nargs="*", default=[],
        help="고정 안전계수 후보 (그리드 위치 0~1). 탐색 없이 그대로 쓴다",
    )
    parser.add_argument(
        "--oracle-bases", nargs="*", default=[],
        help="진단용 반칙 상한 — 홀드아웃을 보고 안전계수를 고른다",
    )
    parser.add_argument(
        "--only-risk", action="store_true",
        help="레거시 maxscore/trust 비교군을 건너뛰고 all_light + 위험 규칙만 돌린다",
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
    print("최종 판정 — 비용 추정 방식별 홀드아웃 성능")
    print(
        f"  분할: 무작위 {args.repeats}회 + 층화 8방향"
        f" · 안전계수 그리드 {args.grid_size}"
    )
    print("  판정 기준: 예산 초과 확률 ≤ 기준선 AND 기대점수 > 기준선")
    print("=" * 78)

    vocab = train_vocabulary(ROOT / "data" / "materialized" / "train" / "inputs.json")
    plans = build_splits(
        inputs, repeats=args.repeats, seed=args.seed, vocab=vocab
    )

    print("  비용 추정기 적합 중...")
    estimators = build_cost_estimators(
        artifact, args.tau, l2=args.l2, iterations=args.iterations,
        learning_rate=args.learning_rate, seed=args.seed,
    )

    score_heads: Dict[str, Any] = {}
    if args.score_heads:
        from analysis.score_heads import build_score_heads

        print("  점수 헤드 적합 중...")
        # 문법: <kind><alpha>[@<bins>]  예) ridge1000, ridge1000@1024
        parsed = []
        for head in args.score_heads:
            spec, _, bins_text = head.partition("@")
            kind = (
                "shrinkt" if spec.startswith("shrinkt")
                else "interact" if spec.startswith("interact")
                else "sparse" if spec.startswith("sparse")
                else "shrink" if spec.startswith("shrink")
                else "boost" if spec.startswith("boost")
                else "light" if spec.startswith("light")
                else "ridge" if spec.startswith("ridge") else "binomial"
            )
            if kind == "sparse":
                value = tuple(spec[len(kind):].split("|"))
            elif kind in ("shrink", "shrinkt"):
                value = tuple(float(v) for v in spec[len(kind):].split(":"))
            else:
                raw_value = float(spec[len(kind):])
                # boost 는 정규화 계수가 아니라 혼합 비중이라 나누지 않는다.
                value = raw_value if kind == "boost" else raw_value / 1760.0
            parsed.append(
                (head, kind, value, int(bins_text) if bins_text else None)
            )
        for bins in sorted({row[3] for row in parsed}, key=lambda b: (b is None, b)):
            group = [row for row in parsed if row[3] == bins]
            raw = build_score_heads(
                artifact,
                ridge_l2=tuple(v for _h, k, v, _b in group if k == "ridge"),
                binomial_l2=tuple(v for _h, k, v, _b in group if k == "binomial"),
                light_l2=tuple(v for _h, k, v, _b in group if k == "light"),
                boost=tuple(v for _h, k, v, _b in group if k == "boost"),
                shrink=tuple(v for _h, k, v, _b in group if k == "shrink"),
                sparse_uplift=tuple(v for _h, k, v, _b in group if k == "sparse"),
                interact=tuple(v * 1760.0 for _h, k, v, _b in group if k == "interact"),
                shrink_train=tuple(v for _h, k, v, _b in group if k == "shrinkt"),
                bins=bins,
            )
            for head, kind, value, _b in group:
                if kind == "interact":
                    label = f"{value * 1760.0:g}"
                elif kind == "sparse":
                    label = "|".join(value)
                elif kind in ("shrink", "shrinkt"):
                    label = ":".join(f"{v:g}" for v in value)
                else:
                    label = f"{value:g}"
                score_heads[head] = raw[f"score_{kind}{label}"]

    # (표시 이름, 비용 추정치, 안전계수 정책)
    methods: List[Tuple[str, Any, str]] = []
    if not args.only_risk:
        # 레거시 비교군 — 이미 기각된 방식들이라 좁은 실험에서는 건너뛴다.
        methods.append(("shipped", None, "maxscore"))
        for name in ("ridge_log_exp", "ridge_smeared"):
            methods.append((name, estimators[name], "maxscore"))
        for tau in args.tau:
            key = f"quantile_{tau:g}"
            methods.append((key, estimators[key], "maxscore"))
            methods.append((f"{key}+trust", estimators[key], "trust"))
        methods.append(("ridge_smeared+trust", estimators["ridge_smeared"], "trust"))
    methods.append(("all_light", None, "all_light"))
    for base in args.oracle_bases:
        costs = None if base == "shipped" else estimators[base]
        methods.append((f"{base}@oracle", costs, "oracle"))
    for fraction in args.fixed_fracs:
        for base in args.risk_bases:
            costs = None if base == "shipped" else estimators[base]
            methods.append((f"{base}@fix{fraction:g}", costs, f"gridfrac{fraction:g}"))
    for spec in args.fixed_tier_fracs:
        for base in args.risk_bases:
            costs = None if base == "shipped" else estimators[base]
            methods.append((f"{base}@fix[{spec}]", costs, f"gridfrac{spec}"))
    for rule in args.risk_rules:
        for base in args.risk_bases:
            costs = None if base == "shipped" else estimators[base]
            methods.append((f"{base}@{rule}", costs, rule))
    # 비용 헤드 alpha 변형 — 점수 alpha 는 쓸었지만 비용은 baseline 기본값(100)을
    # 한 번도 건드리지 않았다. 비용 참값 상한이 +0.0037 로 작아도, 정규화는 정확도가
    # 아니라 "겸손"을 사므로 배분에 영향이 있을 수 있다(alpha=1000 이 이긴 기전).
    for value in args.cost_alphas:
        numpy = _numpy()
        xtr, ytr, _ids = build_matrix("train", artifact)
        xdv, _y, _i = build_matrix("dev", artifact)
        smeared = {}
        for model_id in MODELS:
            target = numpy.log(numpy.maximum(ytr[model_id], 1e-12))
            weights = fit_least_squares(xtr, target, l2=value / xtr.shape[0])
            pred_dev = numpy.exp(numpy.clip(xdv @ weights, -50.0, 50.0))
            pred_train = numpy.exp(numpy.clip(xtr @ weights, -50.0, 50.0))
            factor = float(
                numpy.mean(ytr[model_id] / numpy.maximum(pred_train, 1e-12))
            )
            smeared[model_id] = pred_dev * factor
        for rule in args.risk_rules:
            methods.append((f"costa{value:g}@{rule}", smeared, rule))

    splits = flatten_splits(plans)

    print("  예측 조립 중...")
    _CTX["inputs"] = inputs
    _CTX["outcomes"] = outcomes
    _CTX["policy"] = policy
    _CTX["grid_size"] = args.grid_size
    _CTX["splits"] = splits
    _CTX["calib_resamples"] = args.calib_resamples
    _CTX["seed"] = args.seed
    truth_table = None
    if args.truth:
        from analysis.fast_score import outcome_table

        full = outcome_table(inputs, outcomes, policy)
        truth_table = {}
        if "score" in args.truth:
            # --truth-models 로 일부 모델만 참값으로 바꿀 수 있다. 어느 모델의 점수
            # 예측이 병목인지 분리하기 위한 것이다.
            picked = args.truth_models or list(full["quality"])
            truth_table["quality"] = {
                m: full["quality"][m] for m in picked
            }
        if "cost" in args.truth:
            truth_table["cost"] = full["cost"]
        print(f"  ⚠ 반칙 입력 사용: {'/'.join(args.truth)} 참값 (진단 전용)")

    expanded: List[Tuple[str, Any, str]] = []
    base_methods = list(methods)
    for name, costs, safety_policy in methods:
        expanded.append((name, costs, safety_policy, None))
        for head, values in score_heads.items():
            expanded.append((f"{name}+{head}", costs, safety_policy, values))

    per_tier_specs = []
    for spec in args.tier_heads:
        # 헤드 이름 자체에 콜론이 있을 수 있어(shrink0.85:1:1) 등급 구분자는 '/' 다.
        parts = spec.split("/")
        if len(parts) != 3:
            raise SystemExit(f"--tier-heads 는 fast/balanced/premium 3개 필요: {spec}")
        missing = [h for h in parts if h not in score_heads]
        if missing:
            raise SystemExit(f"--score-heads 에 없는 헤드: {missing}")
        per_tier_specs.append((spec, parts))

    # 등급별 헤드는 **기반 방식**에만 붙인다. 헤드 변형까지 곱하면 조합이 폭증하고
    # 비교 후보만 늘어난다.
    for spec, _parts in per_tier_specs:
        for name, costs, safety_policy in base_methods:
            if safety_policy == "all_light":
                continue
            expanded.append(
                (f"{name}+tier[{spec}]", costs, safety_policy, "PERTIER:" + spec)
            )
    methods = [(n, c, p) for n, c, p, _s in expanded]

    _CTX["methods"] = [
        (
            assemble_predictions(
                inputs.episodes, artifact, costs, truth_table,
                None if isinstance(override, str) else override,
            ),
            safety_policy,
        )
        for _name, costs, safety_policy, override in expanded
    ]
    _CTX["per_tier"] = {}
    for index, (_name, costs, _policy, override) in enumerate(expanded):
        if not isinstance(override, str):
            continue
        spec = override[len("PERTIER:"):]
        parts = dict(zip(TIERS, spec.split("/")))
        _CTX["per_tier"][index] = {
            tier: assemble_predictions(
                inputs.episodes, artifact, costs, truth_table, score_heads[parts[tier]]
            )
            for tier in TIERS
        }

    method_names = [name for name, _costs, _policy in methods]
    jobs = max(1, args.jobs)
    print(
        f"  채점 중: 방식 {len(methods)} × 분할 {len(splits)}"
        f" = {len(methods) * len(splits)} 회 (병렬 {jobs})"
    )
    results = run_judgments(method_names, splits, jobs=jobs)

    report(results)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.report}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
