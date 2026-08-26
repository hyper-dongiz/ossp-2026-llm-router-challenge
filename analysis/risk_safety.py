# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""위험 인지 안전계수 선택 — 보정셋 점수 대신 보정셋 기대점수로 고른다.

baseline 의 `select_safety` 는 보정셋 **점 추정 점수**를 최대화한다. 그 규칙이
hash-regex 를 Premium 3.985 로 몰고 갔고 채점셋에서 4.2 로 터졌다. 점 추정에는
"이 설정이 얼마나 자주 예산을 넘기는가"가 들어 있지 않기 때문이다.

여기서는 같은 보정셋을 복원추출해 **기대점수**(초과 재표본에 0 적용)로 고른다.
자주 터지는 설정은 스스로 낮은 점수를 받아 탈락한다.

## 비용

기대점수는 재표본을 돌리지 않고 `fast_score.moment_stats` 의 닫힌 형태로 얻는다.
후보 25개 × 등급 3 에 0.009초다 — 배분기(`apply_policy`, 후보 25개당 1.39초)에 비하면
무시할 수 있다. 즉 이 교체는 사실상 공짜다.

닫힌 형태를 쓰는 두 번째 이유가 더 중요하다: **numpy 를 쓰지 않아 fork 자식에서
안전하다.** macOS 의 Accelerate BLAS 는 fork 후 교착하므로, 병렬 워커 안에서
numpy 로 부트스트랩을 돌리면 실행이 멈춘다(실제로 겪었다).

## 판정 규칙 (사전 고정)

    expected     보정셋 기대점수 최대
    risk<target> 기대점수 최대 + 초과확률 ≤ target% 제약.
                 제약을 만족하는 후보가 없으면 초과확률이 가장 낮은 후보로 물러선다
                 (실현 불가능할 때 조용히 위험한 값을 고르지 않기 위해서다)

동점은 초과확률이 낮은 쪽, 그다음 안전계수가 낮은 쪽으로 깬다.
"""

from __future__ import annotations

import sys
from math import exp, lgamma, log, log1p
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.fast_score import moment_stats, vectors  # noqa: E402
from analysis.holdout_protocol import _safety_grid, apply_policy  # noqa: E402


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """이항 비율의 보수적 신뢰상한. 관측 0회여도 상한은 0이 아니다."""

    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0

    log_choose = [
        lgamma(trials + 1) - lgamma(i + 1) - lgamma(trials - i + 1)
        for i in range(successes + 1)
    ]

    def cdf(probability: float) -> float:
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
    for _ in range(120):
        mid = (low + high) / 2.0
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def safety_curve(
    predictions: Sequence[Any],
    policy: Any,
    tier: str,
    grid_size: int,
    table: Mapping[str, Any],
    sd_inflate: float = 1.0,
) -> list:
    """안전계수 그리드 전 구간의 (안전계수, 부트스트랩 통계) 곡선.

    곡선을 만들어두면 판정 규칙을 바꿔도 배분기를 다시 돌리지 않는다 — 비싼 계산은
    목표값과 무관하기 때문이다.
    """

    multiplier = table["multiplier"][tier]
    curve = []
    for safety in _safety_grid(policy, tier, grid_size):
        selected = apply_policy(predictions, policy, tier, safety)
        picked_cost, light, picked_quality = vectors(table, selected)
        stats = moment_stats(
            picked_cost, light, picked_quality, multiplier, sd_inflate
        )
        curve.append((safety, stats))
    return curve


def parse_rule(rule: str, tier: str | None = None) -> Tuple[str, float]:
    """규칙 문자열을 (기본 규칙, 분산 팽창 계수) 로 나눈다.

    "expected"          -> ("expected", 1.0)
    "expected1.41"      -> ("expected", 1.41)
    "expected1.41:1:1"  -> 등급별 팽창 (fast:balanced:premium)

    등급별을 두는 이유: 예산 여유가 등급마다 다르다. fast 는 배수 1.25 로 여유가
    25% 뿐이라 보수성이 필요하지만, premium 은 4.0 이라 같은 계수를 걸면 과보수가 된다.
    """

    if ":" in rule:
        head, _, rest = rule.partition(":")
        parts = [head[len("expected"):]] + rest.split(":")
        if len(parts) != 3:
            raise ValueError(f"등급별 팽창은 3개 필요: {rule}")
        order = {"fast": 0, "balanced": 1, "premium": 2}
        return "expected", float(parts[order[tier]] or 1.0)

    for prefix in ("expected", "risk"):
        if rule.startswith(prefix):
            rest = rule[len(prefix):]
            if prefix == "risk":
                # risk20x1.41 형태: 목표% 뒤에 x 로 팽창계수
                target, _, inflate = rest.partition("x")
                return f"risk{target}", float(inflate) if inflate else 1.0
            return prefix, float(rest) if rest else 1.0
    raise ValueError(f"알 수 없는 규칙: {rule}")


def pick_from_curve(curve: Sequence[Tuple[float, Mapping[str, Any]]], rule: str) -> float:
    """곡선에서 규칙에 따라 안전계수 하나를 고른다."""

    if rule == "expected":
        best = max(
            curve,
            key=lambda row: (
                row[1]["expected_score"],
                -row[1]["overrun_rate"],
                -row[0],
            ),
        )
        return best[0]

    if rule.startswith("risk"):
        target = float(rule[4:]) / 100.0
        feasible = [row for row in curve if row[1]["overrun_rate"] <= target]
        if feasible:
            best = max(
                feasible,
                key=lambda row: (
                    row[1]["expected_score"],
                    -row[1]["overrun_rate"],
                    -row[0],
                ),
            )
            return best[0]
        # 실현 불가능 — 조용히 위험한 값을 고르지 않고 초과확률 최소점으로 물러선다.
        return min(curve, key=lambda row: (row[1]["overrun_rate"], row[0]))[0]

    raise ValueError(f"알 수 없는 규칙: {rule}")


def select_safety_risk(
    predictions: Sequence[Any],
    policy: Any,
    tier: str,
    grid_size: int,
    table: Mapping[str, Any],
    rule: str,
) -> Tuple[float, Dict[str, Any]]:
    """곡선을 만들고 규칙으로 고른다."""

    base_rule, inflate = parse_rule(rule, tier)
    curve = safety_curve(predictions, policy, tier, grid_size, table, inflate)
    safety = pick_from_curve(curve, base_rule)
    chosen = next(stats for value, stats in curve if value == safety)
    return safety, dict(chosen)
