# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""검출 검증 — 우리 판정 절차가 '이미 실패한 게 알려진 정책'을 걸러내는가.

측정 도구를 믿으려면 정답을 아는 사례에 걸어봐야 한다. 공개 hash-regex 기본 설정이
그 사례다.

    공개 Dev 880 (in-sample)   Premium 비용비율 3.985 / 한도 4.0 → 통과처럼 보임
    비공개 채점셋              약 4.2 → 한도 초과 → Premium 0점

출처: `baselines/README.md`. Dev 수치는 이 스크립트가 재현한다.

즉 in-sample 채점은 이 정책을 통과로 판정한다. 우리 절차가 **위험**으로 판정하면
검출 성공이고, 통과로 판정하면 절차가 이 실패 유형을 못 잡는다는 뜻이다 — 그렇다면
같은 절차로 낸 다른 결론도 전부 다시 봐야 한다.

## 왜 분할이 아니라 복원추출인가

이 운용점은 안전계수가 아티팩트에 **고정**돼 있다. 보정할 것이 없으므로 보정셋/홀드아웃
분할 자체가 불필요하고, 분할을 없애면 복원추출 중복이 양쪽에 걸치는 누출도 사라진다.
그리고 선택이 없으므로 신뢰상한이 보정이 아니라 **증명서**다.

## 사용

    .venv-train/bin/python analysis/detection_check.py --resamples 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from math import exp, lgamma, log, log1p
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.episode_bootstrap import resample_episodes  # noqa: E402
from analysis.final_judgment import assemble_predictions  # noqa: E402
from analysis.holdout_protocol import apply_policy, score_tier  # noqa: E402
from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

# 공식 등급별 비용 한도 (초과 시 그 등급 0점)
TIER_LIMIT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}

_CTX: Dict[str, Any] = {}


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """이항 비율의 Clopper-Pearson 상한. scipy 없이 CDF 이분탐색으로 푼다.

    관측 빈도가 0이어도 상한은 0이 아니다 — 시행 수가 유한하면 '한 번도 안 나왔다'가
    '일어나지 않는다'를 뜻하지 않는다.
    """

    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0

    log_choose = [
        lgamma(trials + 1) - lgamma(i + 1) - lgamma(trials - i + 1)
        for i in range(successes + 1)
    ]

    def cdf(probability: float) -> float:
        """P(X <= successes). 이항계수가 float 범위를 넘으므로 로그 공간에서 더한다."""

        if probability <= 0.0:
            return 1.0
        if probability >= 1.0:
            return 0.0
        log_p = log(probability)
        log_q = log1p(-probability)
        terms = [
            log_choose[i] + i * log_p + (trials - i) * log_q
            for i in range(successes + 1)
        ]
        peak = max(terms)
        return exp(peak) * sum(exp(t - peak) for t in terms)

    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def score_fixed_point(inputs: Any, outcomes: Any) -> Dict[str, Dict[str, float]]:
    """고정 안전계수로 등급별 비용비율·점수를 낸다 (탐색 없음)."""

    policy = _CTX["policy"]
    safety = _CTX["safety"]
    predictions = _CTX["predictions"]

    out: Dict[str, Dict[str, float]] = {}
    for tier in TIERS:
        selected = apply_policy(predictions, policy, tier, safety[tier])
        report_row = score_tier(inputs, outcomes, policy, tier, selected)
        out[tier] = {
            "budget_ratio": float(Decimal(report_row["budget_ratio"])),
            "budget_passed": bool(report_row["budget_passed"]),
            "tier_score": float(Decimal(report_row["tier_score"])),
        }
    return out


def audit_resample(index: int) -> Dict[str, Dict[str, float]]:
    """복제표본 하나를 고정 운용점으로 채점한다."""

    inputs, outcomes, picks = resample_episodes(
        _CTX["base_inputs"], _CTX["base_outcomes"], _CTX["seed"] + index
    )
    # 예측도 같은 순서로 다시 뽑아야 문항과 짝이 맞는다.
    base = _CTX["base_predictions"]
    _CTX["predictions"] = [base[i] for i in picks]
    try:
        return score_fixed_point(inputs, outcomes)
    finally:
        _CTX["predictions"] = base


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detection-check",
        description="판정 절차가 알려진 실패 정책을 검출하는지 확인합니다.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "baselines" / "hash-regex-public.v1.json",
    )
    parser.add_argument("--resamples", type=int, default=2000)
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
    predictions = assemble_predictions(inputs.episodes, artifact, None)
    safety = {tier: float(artifact.tier_safety_ratios[tier]) for tier in TIERS}

    _CTX.update(
        base_inputs=inputs,
        base_outcomes=outcomes,
        base_predictions=predictions,
        predictions=predictions,
        policy=policy,
        safety=safety,
        seed=args.seed,
    )

    print("=" * 78)
    print("검출 검증 — 공개 hash-regex 기본 설정")
    print(f"  고정 안전계수: " + " · ".join(f"{t} {safety[t]:.4f}" for t in TIERS))
    print(f"  복원추출 {args.resamples}회 (분할 없음 — 보정할 것이 없다)")
    print("=" * 78)

    in_sample = score_fixed_point(inputs, outcomes)

    print("\n[1] in-sample — 공개 Dev 880 전체를 그대로 채점 (팀들이 속는 지점)")
    print(
        f"  {'등급':<10}{'비용비율':>10}{'한도':>8}{'여유':>9}{'통과':>7}{'점수':>9}"
    )
    for tier in TIERS:
        row = in_sample[tier]
        limit = TIER_LIMIT[tier]
        margin = (limit - row["budget_ratio"]) / limit
        print(
            f"  {tier:<10}{row['budget_ratio']:>10.4f}{limit:>8.2f}"
            f"{margin*100:>8.2f}%{('통과' if row['budget_passed'] else '초과'):>7}"
            f"{row['tier_score']:>9.4f}"
        )

    jobs = max(1, args.jobs)
    print(f"\n  감사 중: 복제표본 {args.resamples}개 (병렬 {jobs})")
    results: List[Any] = []
    if jobs > 1:
        import multiprocessing

        context = multiprocessing.get_context("fork")
        with context.Pool(processes=jobs) as pool:
            for done, row in enumerate(
                pool.imap_unordered(
                    audit_resample, range(args.resamples), chunksize=8
                ),
                start=1,
            ):
                results.append(row)
                if done % 250 == 0 or done == args.resamples:
                    print(f"    {done}/{args.resamples}", flush=True)
    else:
        for index in range(args.resamples):
            results.append(audit_resample(index))

    audit: Dict[str, Any] = {}
    print("\n[2] 감사 — 문항을 다시 뽑았을 때 (선택 없음 → 상한이 증명서)")
    print(
        f"  {'등급':<10}{'초과횟수':>12}{'빈도':>9}{'95% 상한':>10}"
        f"{'평균비율':>10}{'최악비율':>10}"
    )
    for tier in TIERS:
        ratios = [row[tier]["budget_ratio"] for row in results]
        overruns = sum(1 for row in results if not row[tier]["budget_passed"])
        upper = clopper_pearson_upper(overruns, len(results))
        ordered = sorted(ratios)
        audit[tier] = {
            "overruns": overruns,
            "trials": len(results),
            "overrun_rate": overruns / len(results),
            "clopper_pearson_upper": upper,
            "mean_ratio": sum(ratios) / len(ratios),
            "worst_ratio": max(ratios),
            "ratio_percentiles": {
                str(q): ordered[min(len(ordered) - 1, int(q / 100 * len(ordered)))]
                for q in (5, 25, 50, 75, 95, 99)
            },
            # 알려진 채점셋 관측값이 이 분포의 몇 분위인가 (Premium 만 관측치 존재)
            "ratios": ordered,
        }
        print(
            f"  {tier:<10}{f'{overruns}/{len(results)}':>12}"
            f"{overruns/len(results)*100:>8.2f}%{upper*100:>9.2f}%"
            f"{audit[tier]['mean_ratio']:>10.4f}{audit[tier]['worst_ratio']:>10.4f}"
        )

    # 판정 — in-sample 이 통과라고 한 등급 중, 감사가 실질 위험을 지적한 것이 있는가.
    print("\n[3] 판정")
    detected: List[str] = []
    for tier in TIERS:
        if in_sample[tier]["budget_passed"] and audit[tier]["overrun_rate"] > 0.05:
            detected.append(tier)
    for tier in TIERS:
        mark = "검출" if tier in detected else "—"
        print(
            f"  {tier:<10}in-sample "
            f"{'통과' if in_sample[tier]['budget_passed'] else '초과'}"
            f" → 감사 초과확률 {audit[tier]['overrun_rate']*100:.1f}%"
            f"  [{mark}]"
        )
    verdict = "premium" in detected
    print()
    if verdict:
        print("  ✅ 검출 성공 — in-sample 이 통과로 본 Premium 을 절차가 위험으로 판정.")
        print("     알려진 실패(채점셋 약 4.2 → 0점)와 같은 방향이다.")
    else:
        print("  ❌ 검출 실패 — 절차가 알려진 실패를 못 잡는다.")
        print("     이 절차로 낸 다른 결론도 전부 재검토 대상이다.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "report_type": "detection-check-v1",
                    "artifact": str(args.artifact.name),
                    "safety_ratios": safety,
                    "resamples": args.resamples,
                    "in_sample": in_sample,
                    "audit": audit,
                    "detected_tiers": detected,
                    "premium_detected": verdict,
                },
                ensure_ascii=False,
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.report}")
    print()
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
