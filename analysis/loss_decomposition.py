# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""예측 손실을 비용 예측과 점수 예측으로 분해한다.

## 배경

`final_judgment.py` 결과로 손실이 둘로 나뉜다는 것이 확인됐다.

    0.8034  참값으로 배낭을 풀었을 때의 천장
    0.6927  현재 예측으로, 한 번도 예산을 안 넘겼다면
    0.6193  all-light (위험 0)
    0.4060  현재 실제 기대점수

    위험 손실 0.287 = 0.6927 - 0.4060   ← 트랙 A (배분·위험 통제)
    예측 손실 0.111 = 0.8034 - 0.6927   ← 트랙 B (추정)  ← 이 스크립트

트랙 B 안에서 비용 예측과 점수 예측 중 어느 쪽이 손실을 더 만드는지 모르면
어디에 시간을 쓸지 정할 수 없다.

## 방법

배분기(라그랑주 + 이분탐색)와 안전계수(1.0 고정)를 붙잡고, 배분기에 들어가는
두 입력을 참값/예측값으로 교차 대입해 채점한다.

    참점수 + 참비용    천장
    참점수 + 예측비용  비용 예측만 틀렸을 때
    예측점수 + 참비용  점수 예측만 틀렸을 때
    예측점수 + 예측비용 둘 다 틀렸을 때 (= 현재)

안전계수를 1.0으로 고정하는 이유: 안전계수 탐색은 트랙 A의 변수이므로
여기서 섞으면 원인이 흐려진다. 예산 초과분은 그대로 0점 처리해 실제 규칙을 따른다.

주의: 참값을 배분에 쓰는 것은 상한 측정용이며 제출에는 쓸 수 없다
(라우터는 평가 결과를 입력으로 받지 않는다).
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.holdout_protocol import apply_policy, score_tier  # noqa: E402
from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

MODELS = ("ax31-light", "ax31", "axk1-think")
RATES = {
    "ax31-light": (Decimal("1"), Decimal("4")),
    "ax31": (Decimal("2.127"), Decimal("8.509")),
    "axk1-think": (Decimal("6.565"), Decimal("26.260")),
}
TOKEN_UNIT = Decimal("1000000")
TIER_WEIGHT = {
    "fast": Decimal("0.4"),
    "balanced": Decimal("0.3"),
    "premium": Decimal("0.3"),
}


def true_values(split: str, episodes: Sequence[Any]) -> tuple:
    raw = json.loads((ROOT / "data" / split / "outcomes.json").read_text("utf-8"))
    models = {row["episode_id"]: row["models"] for row in raw["episodes"]}
    scores: List[Dict[str, float]] = []
    costs: List[Dict[str, float]] = []
    for episode in episodes:
        entry = models[episode.episode_id]
        scores.append(
            {m: float(Decimal(entry[m]["score"])) for m in MODELS}
        )
        costs.append(
            {
                m: float(
                    (
                        Decimal(entry[m]["input_tokens"]) * RATES[m][0]
                        + Decimal(entry[m]["output_tokens"]) * RATES[m][1]
                    )
                    / TOKEN_UNIT
                )
                for m in MODELS
            }
        )
    return scores, costs


def predicted_values(episodes: Sequence[Any], artifact: Any) -> tuple:
    scores: List[Mapping[str, float]] = []
    costs: List[Mapping[str, float]] = []
    for episode in episodes:
        s, c = hash_regex.predict_episode(episode, artifact)
        scores.append(s)
        costs.append(c)
    return scores, costs


def run_combination(
    label: str,
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    inputs: Any,
    outcomes: Any,
    policy: Any,
) -> Mapping[str, Any]:
    entry: Dict[str, Any] = {"label": label, "tiers": {}}
    weighted = Decimal(0)
    for tier in TIERS:
        predictions = list(zip(scores, costs))
        selected = apply_policy(predictions, policy, tier, 1.0)
        report = score_tier(inputs, outcomes, policy, tier, selected)
        raw = Decimal(report["tier_score"])
        passed = bool(report["budget_passed"])
        entry["tiers"][tier] = {
            "raw_score": float(raw),
            "budget_ratio": float(Decimal(report["budget_ratio"])),
            "budget_multiplier": float(policy.tiers[tier].budget_multiplier),
            "budget_passed": passed,
            "effective_score": float(raw if passed else Decimal(0)),
        }
        weighted += TIER_WEIGHT[tier] * (raw if passed else Decimal(0))
    entry["weighted_effective"] = float(weighted)
    entry["weighted_raw"] = float(
        sum(
            TIER_WEIGHT[t] * Decimal(str(entry["tiers"][t]["raw_score"]))
            for t in TIERS
        )
    )
    return entry


def report(rows: Sequence[Mapping[str, Any]]) -> None:
    width = max(len(r["label"]) for r in rows) + 2
    print("\n[1] 등급별 점수 / 비용비율 / 통과")
    for tier in TIERS:
        print(f"  ── {tier}  (한도 {rows[0]['tiers'][tier]['budget_multiplier']})")
        print(f"     {'조합':<{width}}{'점수':>9}{'비용비율':>10}{'통과':>7}")
        for r in rows:
            t = r["tiers"][tier]
            print(
                f"     {r['label']:<{width}}{t['raw_score']:>9.4f}"
                f"{t['budget_ratio']:>10.3f}{'O' if t['budget_passed'] else 'X':>7}"
            )

    print("\n[2] 가중 최종 — 초과 시 0 적용")
    print(f"  {'조합':<{width}}{'기대점수':>10}{'초과무시':>10}")
    for r in rows:
        print(
            f"  {r['label']:<{width}}{r['weighted_effective']:>10.4f}"
            f"{r['weighted_raw']:>10.4f}"
        )

    ceiling = rows[0]["weighted_raw"]
    print("\n[3] 손실 분해 (초과 무시 기준 — 위험 손실을 분리해 보기 위함)")
    for r in rows[1:]:
        print(
            f"  {r['label']:<{width}}천장 대비 {ceiling - r['weighted_raw']:+.4f}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loss-decomposition",
        description="예측 손실을 비용/점수로 분해합니다.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "baselines" / "hash-regex-public.v1.json",
    )
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = hash_regex.load_artifact(args.artifact)
    policy = load_bundled_policy()
    inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")

    print("=" * 74)
    print("예측 손실 분해 — 안전계수 1.0 고정, 배분기 동일")
    print("=" * 74)

    true_scores, true_costs = true_values("dev", inputs.episodes)
    pred_scores, pred_costs = predicted_values(inputs.episodes, artifact)

    rows = [
        run_combination("참점수 + 참비용 (천장)", true_scores, true_costs, inputs, outcomes, policy),
        run_combination("참점수 + 예측비용", true_scores, pred_costs, inputs, outcomes, policy),
        run_combination("예측점수 + 참비용", pred_scores, true_costs, inputs, outcomes, policy),
        run_combination("예측점수 + 예측비용 (현재)", pred_scores, pred_costs, inputs, outcomes, policy),
    ]
    report(rows)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(rows, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.report}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
