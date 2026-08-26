# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""목적함수를 바꾸면 어느 운용점이 최적인가 — 기댓값 대신 순위.

지금까지 기대점수 E[score] 를 최대화해왔다. 제출이 단발이면 그것이 옳은 목적함수다.
그런데 **순위 경쟁은 다른 게임**이다. 40개 팀 중 위험을 감수한 팀이 여럿이면 그중
몇은 운이 좋아 높은 점수에 안착하고, 우리는 절대 거기 못 간다 — 우리 분포의 천장이
낮기 때문이다.

실측이 그렇다. 우리 최종 정책은 hash-regex 원본 규칙과 1:1 로 붙으면 67% 만 이긴다.
상대의 95분위(0.7014)가 우리 최댓값(0.6977)보다 높다.

## 무엇을 계산하는가

기존 판정 리포트에 분할별 점수가 남아 있으므로 **재실행 없이** 목적함수만 바꿔
다시 순위를 매길 수 있다. 세 가지를 낸다.

    E[score]          기댓값 (지금까지의 목적)
    P(score >= t)     상위권 진입 확률. t 는 여러 값으로 훑는다
    P(우리 > 상대)     1:1 승률. 같은 분할끼리 짝지어 센다

그리고 **N명 중 1등 확률**을 근사한다. 경쟁자 N 명이 각각 상대 분포에서 독립으로
뽑는다고 보고, 우리 점수가 그 최댓값을 넘을 확률을 분할별로 계산해 평균낸다.
경쟁자 분포는 리포트에 있는 실제 방식 중 하나를 골라 쓴다(기본은 hash-regex 원본).

## 한계

경쟁자가 실제로 어떤 분포를 쓰는지 모른다. 여기서 쓰는 것은 "공개 baseline 규칙을
그대로 쓰는 팀"의 분포이고, 진지한 팀은 그보다 낫다. 그래서 이 수치는 **순위 목적이
기댓값 목적과 얼마나 다른 답을 주는지**를 보는 용도이지 실제 우승 확률의 추정이 아니다.

## 사용

    python3 analysis/win_probability.py \\
        --reports analysis/reports/final-judgment-round15.json ... \\
        --rivals 40
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_methods(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """여러 리포트에서 방식별 분할 점수를 모은다.

    분할은 seed 로 결정적이라 실행이 달라도 같은 순서다 — 같은 이름이 여러 리포트에
    있으면 첫 번째만 쓰고, 분할 이름이 다르면 섞지 않는다.
    """

    methods: Dict[str, Dict[str, Any]] = {}
    reference: List[str] | None = None
    for path in paths:
        for row in json.loads(path.read_text("utf-8")):
            names = row.get("split_names")
            if names is None:
                continue
            if reference is None:
                reference = names
            elif names != reference:
                raise SystemExit(f"분할 구성이 다른 리포트: {path}")
            methods.setdefault(
                row["method"],
                {
                    "scores": [
                        v
                        for v, n in zip(row["per_split_weighted"], names)
                        if n == "random"
                    ],
                    "source": path.name,
                },
            )
    return methods


def win_rate(ours: Sequence[float], rival: Sequence[float]) -> float:
    """같은 분할끼리 짝지어 우리가 이긴 비율."""

    return statistics.fmean(1.0 if a > b else 0.0 for a, b in zip(ours, rival))


def top_rate(ours: Sequence[float], rival: Sequence[float], rivals: int) -> float:
    """경쟁자 N 명 중 1등 확률 (근사).

    분할 하나를 "한 번의 채점"으로 보고, 경쟁자 N 명이 상대 분포에서 독립으로 뽑는다고
    가정한다. 우리 점수가 N 명 전원을 넘을 확률은 (상대 분포에서 우리보다 낮을 확률)^N
    이므로, 상대 표본의 경험분포로 그 값을 낸다.
    """

    ordered = sorted(rival)
    count = len(ordered)
    total = 0.0
    for value in ours:
        below = sum(1 for r in ordered if r < value) / count
        total += below**rivals
    return total / len(ours)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="win-probability")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--rival", default="shipped", help="경쟁자로 삼을 방식 이름")
    parser.add_argument("--rivals", type=int, default=40)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.68, 0.69, 0.70],
    )
    args = parser.parse_args(argv)

    methods = load_methods(args.reports)
    if args.rival not in methods:
        raise SystemExit(
            f"경쟁자 방식 '{args.rival}' 이 리포트에 없습니다. "
            f"가능: {sorted(methods)[:8]}"
        )
    rival = methods[args.rival]["scores"]

    rows = []
    for name, payload in methods.items():
        ours = payload["scores"]
        rows.append(
            {
                "method": name,
                "expected": statistics.fmean(ours),
                "win": win_rate(ours, rival),
                "top": top_rate(ours, rival, args.rivals),
                "reach": {
                    t: statistics.fmean(1.0 if v >= t else 0.0 for v in ours)
                    for t in args.thresholds
                },
                "p95": sorted(ours)[int(0.95 * (len(ours) - 1))],
            }
        )

    print("=" * 92)
    print(f"목적함수별 순위 — 경쟁자 '{args.rival}' 분포, {args.rivals}명 가정")
    print("=" * 92)
    header = (
        f"\n{'방식':<40}{'기댓값':>9}{'1:1승률':>9}"
        f"{f'1등({args.rivals}명)':>12}{'95분위':>9}"
        + "".join(f"{'P≥'+f'{t:g}':>9}" for t in args.thresholds)
    )
    for key, label in (
        ("expected", "기댓값 최대"),
        ("win", "1:1 승률 최대"),
        ("top", f"{args.rivals}명 중 1등 확률 최대"),
    ):
        print(f"\n── {label} 기준 상위 6")
        print(header.strip("\n"))
        for row in sorted(rows, key=lambda r: -r[key])[:6]:
            reach = "".join(f"{row['reach'][t]*100:>8.0f}%" for t in args.thresholds)
            print(
                f"{row['method'][:38]:<40}{row['expected']:>9.4f}"
                f"{row['win']*100:>8.0f}%{row['top']*100:>11.1f}%"
                f"{row['p95']:>9.4f}{reach}"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
