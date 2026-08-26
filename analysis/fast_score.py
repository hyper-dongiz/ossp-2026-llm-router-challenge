# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""고속 대리 채점기 — 선택이 고정되면 채점은 문항별 합으로 분해된다.

`scoring._score_tier` 를 읽으면 한 등급의 채점은 세 개의 합으로 끝난다.

    총비용        = Σ cost(선택 모델_i)
    light 기준비용 = Σ cost(light_i)
    품질          = Σ score(선택 모델_i) / n
    등급점수      = 예산 통과 시 품질, 초과 시 0
    통과 여부     = 총비용 ≤ light 기준비용 × 등급 배수

즉 선택이 정해지면 문항별 삼중항 (c_i, l_i, q_i) 만 있으면 되고, 부트스트랩 재표본은
가중치 w_i 를 곱한 합으로 바로 계산된다. 공식 채점기를 다시 부를 필요가 없다.

    비율_재표본 = Σ w_i·c_i / Σ w_i·l_i
    점수_재표본 = Σ w_i·q_i / Σ w_i

이 분해 덕분에 재표본 수천 개가 밀리초 단위로 끝난다. 보정셋에서 안전계수를 고를 때
"이 안전계수의 초과확률이 얼마인가"를 실제로 물어볼 수 있게 되는 것이 목적이다.

**공식 채점기와의 일치는 `verify_parity()` 로 확인한다.** 대리 채점기를 쓰는 모든
경로는 이 검증을 통과한 뒤에만 신뢰한다. Decimal 대신 float 를 쓰므로 완전 동일이
아니라 허용 오차 안에서의 일치다 — 선택 판단에는 충분하고, 최종 보고 수치는 항상
공식 채점기로 다시 낸다.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ossp_router.protocol import TIERS  # noqa: E402


def cost_of(outcome: Any, policy: Any) -> float:
    """scoring._cost 와 같은 식. Decimal 대신 float."""

    rates = policy.models[outcome.model_id]
    unit = float(policy.token_unit)
    return (
        float(rates.fixed_cost)
        + outcome.input_tokens * float(rates.input_token_rate) / unit
        + outcome.output_tokens * float(rates.output_token_rate) / unit
    )


def outcome_table(
    inputs: Any, outcomes: Any, policy: Any
) -> Dict[str, Any]:
    """문항 × 모델 별 (비용, 점수) 표와 light 비용 벡터를 만든다.

    한 번 만들어두면 어떤 선택에 대해서도 재사용된다 — 선택이 바뀌어도 문항별
    비용·점수 자체는 변하지 않기 때문이다.
    """

    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    ids = [episode.episode_id for episode in inputs.episodes]
    models = list(policy.models)

    cost = {m: [] for m in models}
    quality = {m: [] for m in models}
    for episode_id in ids:
        for model_id in models:
            row = index[(episode_id, model_id)]
            cost[model_id].append(cost_of(row, policy))
            quality[model_id].append(float(row.score))
    return {
        "ids": ids,
        "models": models,
        "cost": cost,
        "quality": quality,
        "light": list(cost[policy.light_model_id]),
        "multiplier": {t: float(policy.tiers[t].budget_multiplier) for t in TIERS},
    }


def vectors(
    table: Mapping[str, Any], selected: Sequence[str]
) -> Tuple[list, list, list]:
    """선택된 모델 열에서 문항별 (비용, light 비용, 품질) 을 뽑는다."""

    cost = table["cost"]
    quality = table["quality"]
    picked_cost = [cost[model_id][i] for i, model_id in enumerate(selected)]
    picked_quality = [quality[model_id][i] for i, model_id in enumerate(selected)]
    return picked_cost, list(table["light"]), picked_quality


def score_once(
    picked_cost: Sequence[float],
    light_cost: Sequence[float],
    picked_quality: Sequence[float],
    multiplier: float,
) -> Dict[str, float]:
    """재표본 없이 그대로 채점 — 공식 채점기와 대조하는 기준."""

    total = sum(picked_cost)
    base = sum(light_cost)
    ratio = total / base
    passed = total <= base * multiplier
    quality = sum(picked_quality) / len(picked_quality)
    return {
        "budget_ratio": ratio,
        "budget_passed": passed,
        "tier_score": quality if passed else 0.0,
        "quality_score": quality,
    }


def make_weights(count: int, resamples: int, seed: int) -> Any:
    """복원추출 가중치 행렬 (resamples × count).

    안전계수 후보마다 새로 뽑지 않고 **하나를 만들어 돌려쓴다.** 생성이 계산의
    대부분이라 빠르기도 하지만, 그보다 후보들이 같은 재표본을 보게 되어 비교가
    쌍대로 성립하는 것이 더 중요하다 — 재표본 운이 후보 간 차이에서 상쇄된다.
    """

    import numpy

    rng = numpy.random.default_rng(seed)
    return rng.multinomial(count, numpy.full(count, 1.0 / count), size=resamples)


def bootstrap_stats(
    picked_cost: Sequence[float],
    light_cost: Sequence[float],
    picked_quality: Sequence[float],
    multiplier: float,
    *,
    weights: Any,
) -> Dict[str, float]:
    """문항 복원추출로 기대점수와 초과확률을 낸다.

    반환하는 `expected_score` 가 선택 기준이다 — 초과한 재표본에 0을 넣고 평균내므로
    "점수는 높지만 자주 터지는" 설정이 자동으로 낮게 평가된다.
    """

    import numpy

    count = len(picked_cost)
    cost = numpy.asarray(picked_cost, dtype=float)
    light = numpy.asarray(light_cost, dtype=float)
    quality = numpy.asarray(picked_quality, dtype=float)

    totals = weights @ cost
    bases = weights @ light
    qualities = (weights @ quality) / count

    ratios = totals / bases
    passed = totals <= bases * multiplier
    effective = numpy.where(passed, qualities, 0.0)
    return {
        "expected_score": float(effective.mean()),
        "overrun_rate": float(1.0 - passed.mean()),
        "mean_ratio": float(ratios.mean()),
        "worst_ratio": float(ratios.max()),
        "quality_when_passed": (
            float(qualities[passed].mean()) if passed.any() else 0.0
        ),
    }


def verify_parity(
    inputs: Any,
    outcomes: Any,
    policy: Any,
    table: Mapping[str, Any],
    selected_by_tier: Mapping[str, Sequence[str]],
    official: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """공식 채점기 결과와 대리 채점기 결과를 대조한다.

    official 은 tier -> score_tier(...) 반환값. 불일치가 있으면 그대로 담아 돌려주고,
    호출자가 실패로 처리한다.
    """

    problems = []
    detail = {}
    for tier in TIERS:
        picked_cost, light, picked_quality = vectors(table, selected_by_tier[tier])
        fast = score_once(picked_cost, light, picked_quality, table["multiplier"][tier])
        ref = official[tier]
        ratio_gap = abs(fast["budget_ratio"] - float(Decimal(ref["budget_ratio"])))
        score_gap = abs(fast["tier_score"] - float(Decimal(ref["tier_score"])))
        same_pass = fast["budget_passed"] == bool(ref["budget_passed"])
        detail[tier] = {
            "ratio_gap": ratio_gap,
            "score_gap": score_gap,
            "pass_agrees": same_pass,
        }
        if ratio_gap > tolerance or score_gap > tolerance or not same_pass:
            problems.append(tier)
    return {"ok": not problems, "problems": problems, "detail": detail}


# ── 닫힌 형태 추정 — 재표본 없이 모멘트만으로 ────────────────────────────────
#
# 재표본을 실제로 돌리지 않아도 된다. 통과 판정은
#
#     Σ w_i·c_i ≤ m · Σ w_i·l_i   ⟺   Σ w_i·d_i ≤ 0,   d_i = c_i − m·l_i
#
# 하나의 부호 판정으로 줄고, 다항 가중치 합 S = Σ w_i d_i 의 평균·분산은 표본
# 모멘트로 바로 나온다 (w ~ Multinomial(n, 1/n)).
#
#     E[S]   = Σ d_i
#     Var[S] = Σ d_i² − (Σ d_i)² / n
#
# 품질 Q = Σ w_i q_i / n 도 같고, 둘의 공분산까지 있으면 기대점수가 닫힌 식이 된다.
#
#     E[Q·1(S≤0)] = μ_Q·Φ(z) − ρ·σ_Q·φ(z),   z = −μ_S / σ_S
#
# O(n) 두 번이면 끝나므로 재표본 500회 대비 1000배 빠르고, **numpy 를 쓰지 않아
# fork 자식에서 안전하다** (macOS Accelerate 는 fork 후 교착한다).
#
# 정규근사이므로 꼬리 확률에는 부정확하다. 다만 우리 운용 구간의 초과확률은
# 20~50% 로 분포의 중앙이라 근사가 잘 듣는다 — `verify_moment_parity()` 로
# 실제 부트스트랩과 대조해 확인한다.


def moment_stats(
    picked_cost: Sequence[float],
    light_cost: Sequence[float],
    picked_quality: Sequence[float],
    multiplier: float,
    sd_inflate: float = 1.0,
) -> Dict[str, float]:
    """재표본 없이 기대점수·초과확률을 낸다. 순수 파이썬, O(n).

    sd_inflate 는 보정셋→홀드아웃 드리프트 보정이다. 부트스트랩은 **같은 표본 안에서의**
    변동만 재는데 실제로는 홀드아웃이 **다른 표본**이다. 두 반쪽이 모두 무작위이므로
    합쳐진 변동은 대략 √2 배이고, 이 계수가 그 몫을 반영한다. 1.0 이면 보정 없음.
    """

    from math import erf, exp as _exp, pi, sqrt

    count = len(picked_cost)
    sum_d = sum_dd = sum_q = sum_qq = sum_dq = 0.0
    for index in range(count):
        d = picked_cost[index] - multiplier * light_cost[index]
        q = picked_quality[index]
        sum_d += d
        sum_dd += d * d
        sum_q += q
        sum_qq += q * q
        sum_dq += d * q

    mean_s = sum_d
    var_s = sum_dd - sum_d * sum_d / count
    mean_q = sum_q / count
    var_q = (sum_qq - sum_q * sum_q / count) / (count * count)
    cov = (sum_dq - sum_d * sum_q / count) / count

    if var_s <= 0.0:
        # 모든 문항의 여유가 동일 — 통과 여부가 표본과 무관하다.
        passed = mean_s <= 0.0
        return {
            "expected_score": mean_q if passed else 0.0,
            "overrun_rate": 0.0 if passed else 1.0,
            "pass_rate": 1.0 if passed else 0.0,
            "quality_mean": mean_q,
        }

    sd_s = sqrt(var_s) * sd_inflate
    sd_q = sqrt(var_q) if var_q > 0.0 else 0.0
    z = -mean_s / sd_s
    normal_cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    normal_pdf = _exp(-0.5 * z * z) / sqrt(2.0 * pi)
    rho = cov / (sd_s * sd_q) if sd_q > 0.0 else 0.0
    rho = max(-1.0, min(1.0, rho))

    expected = mean_q * normal_cdf - rho * sd_q * normal_pdf
    return {
        "expected_score": max(0.0, expected),
        "overrun_rate": 1.0 - normal_cdf,
        "pass_rate": normal_cdf,
        "quality_mean": mean_q,
    }


def verify_moment_parity(
    picked_cost: Sequence[float],
    light_cost: Sequence[float],
    picked_quality: Sequence[float],
    multiplier: float,
    *,
    weights: Any,
) -> Dict[str, float]:
    """닫힌 형태 추정과 실제 부트스트랩을 대조한다 (부모 프로세스 전용)."""

    exact = bootstrap_stats(
        picked_cost, light_cost, picked_quality, multiplier, weights=weights
    )
    approx = moment_stats(picked_cost, light_cost, picked_quality, multiplier)
    return {
        "bootstrap_expected": exact["expected_score"],
        "moment_expected": approx["expected_score"],
        "expected_gap": abs(exact["expected_score"] - approx["expected_score"]),
        "bootstrap_overrun": exact["overrun_rate"],
        "moment_overrun": approx["overrun_rate"],
        "overrun_gap": abs(exact["overrun_rate"] - approx["overrun_rate"]),
    }
