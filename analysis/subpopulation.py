# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""비용 편향이 몰려 있는 문항군을 프롬프트만으로 식별한다.

배경: 공개 Dev 880문항은 균질하지 않다. 운영자 생성 횟수가 4회인 123문항(14%)이
전체 실제 비용의 27~28%를 쓰면서 예측 편향이 그쪽에 집중된다.

    싼 모델   2회군 편향  7.0%  vs  4회군  64.9%
    비싼 모델 2회군 편향 30.2%  vs  4회군 164.8%

전역 보정계수 하나로는 165% 집단과 30% 집단을 같이 맞출 수 없다.
따라서 (1) 프롬프트만으로 그 집단을 식별하고 (2) 집단별 보정계수를 적용한다.

생성 횟수는 채점 데이터에만 있어 라우터가 볼 수 없다. 학습·설계에는 쓸 수 있으나
(공개 자료는 학습·최적화 허용) 런타임 판별은 프롬프트 내용만으로 해야 한다.

주의: 생성 횟수는 편향의 원인이 아니라 문항 종류를 알려주는 표식일 뿐이다.
비공개 채점셋에 같은 표식이 있을 보장이 없으므로, 목표는 "생성 횟수 맞히기"가
아니라 "비용 편향이 큰 문항 종류 맞히기"다. 지표도 최종 편향 감소로 판단한다.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from baselines import hash_regex  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import Episode, load_input  # noqa: E402

MODELS = ("ax31-light", "ax31", "axk1-think")
LABEL = {"ax31-light": "싼 모델", "ax31": "중간 모델", "axk1-think": "비싼 추론 모델"}

RATES = {
    "ax31-light": (Decimal("1"), Decimal("4")),
    "ax31": (Decimal("2.127"), Decimal("8.509")),
    "axk1-think": (Decimal("6.565"), Decimal("26.260")),
}
TOKEN_UNIT = Decimal("1000000")

# ── 후보 신호 ────────────────────────────────────────────────
# 전부 표준 라이브러리 정규식. 제출 컨테이너에서 그대로 동작한다.

_ASK_QUANTITY = re.compile(r"\bhow (?:many|much)\b", re.IGNORECASE)
_TASK_VERB = re.compile(
    r"\b(?:find|compute|calculate|determine|total|remaining|each)\b", re.IGNORECASE
)
_LATEX = re.compile(r"\\\[|\\\(|\\frac|\\sum|\\sqrt|\\text|\$")
_CODE = re.compile(r"```|\bdef\b|\bimport\b|\bclass\b|#include|[{};]\s*$", re.MULTILINE)
_SYNTHETIC_ALGEBRA = re.compile(r"\*\*|\bLet\s+\w+\s*[(=]|\bSuppose\b")
_HANGUL = re.compile(r"[가-힣]")


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isascii() and c.isalpha() for c in text) / len(text)


SIGNALS: Mapping[str, Callable[[str], bool]] = {
    "ask_quantity": lambda t: bool(_ASK_QUANTITY.search(t)),
    "task_verb": lambda t: bool(_TASK_VERB.search(t)),
    "latex": lambda t: bool(_LATEX.search(t)),
    "latin_heavy": lambda t: latin_ratio(t) >= 0.5,
    "has_digit": lambda t: any(c.isdigit() for c in t),
    "no_hangul": lambda t: not _HANGUL.search(t),
    "no_code": lambda t: not _CODE.search(t),
    "no_synthetic_algebra": lambda t: not _SYNTHETIC_ALGEBRA.search(t),
}


# ── 판별 규칙 ────────────────────────────────────────────────


def rule_numeric_density(text: str) -> bool:
    """기존 특징만 쓴 참조 규칙 (비교용)."""

    nonspace = sum(not c.isspace() for c in text)
    digits = sum(c.isdigit() for c in text)
    return (digits / max(1, nonspace)) >= 0.01


def rule_word_problem(text: str) -> bool:
    """제안 규칙 — 영어 서술형 수학/추론 문항.

    양성 신호(수량 질문·과제 동사·LaTeX) 중 하나 이상이면서,
    음성 신호(한글·코드·합성 대수식)에 걸리지 않아야 한다.
    """

    if _HANGUL.search(text):
        return False
    if _CODE.search(text):
        return False
    if _SYNTHETIC_ALGEBRA.search(text):
        return False
    if latin_ratio(text) < 0.3:
        return False
    return bool(
        _ASK_QUANTITY.search(text) or _TASK_VERB.search(text) or _LATEX.search(text)
    )


RULES: Mapping[str, Callable[[str], bool]] = {
    "numeric_density≥0.01 (기존 특징)": rule_numeric_density,
    "word_problem (제안)": rule_word_problem,
}


# ── 데이터 ──────────────────────────────────────────────────


def load_split(split: str) -> List[Tuple[Episode, Mapping[str, Any], int]]:
    outcomes = {
        row["episode_id"]: row
        for row in json.loads(
            (ROOT / "data" / split / "outcomes.json").read_text("utf-8")
        )["episodes"]
    }
    episodes = load_input(
        ROOT / "data" / "materialized" / split / "inputs.json"
    ).episodes
    return [
        (
            e,
            outcomes[e.episode_id]["models"],
            outcomes[e.episode_id]["models"]["ax31-light"]["num_generations"],
        )
        for e in episodes
    ]


def actual_cost(model_id: str, entry: Mapping[str, Any]) -> float:
    rate_in, rate_out = RATES[model_id]
    return float(
        (
            Decimal(entry["input_tokens"]) * rate_in
            + Decimal(entry["output_tokens"]) * rate_out
        )
        / TOKEN_UNIT
    )


# ── 1. 신호별 분리력 ────────────────────────────────────────


def measure_signals(rows: Sequence[Any]) -> Mapping[str, Any]:
    high = [r for r in rows if r[2] == 4]
    low = [r for r in rows if r[2] == 2]
    out: Dict[str, Any] = {"high_count": len(high), "low_count": len(low), "signals": {}}
    for name, fn in SIGNALS.items():
        a = statistics.fmean(fn(episode_text(r[0])) for r in high)
        b = statistics.fmean(fn(episode_text(r[0])) for r in low)
        out["signals"][name] = {"in_high": a, "in_low": b, "gap": a - b}
    return out


def report_signals(res: Mapping[str, Any]) -> None:
    print(
        f"\n[1] 신호별 분리력  (편향 큰 군 {res['high_count']} / "
        f"작은 군 {res['low_count']})"
    )
    print(f"  {'신호':<22}{'편향큰군':>10}{'편향작은군':>12}{'차이':>9}")
    for name, r in sorted(
        res["signals"].items(), key=lambda kv: -abs(kv[1]["gap"])
    ):
        print(
            f"  {name:<22}{r['in_high']*100:>9.1f}%{r['in_low']*100:>11.1f}%"
            f"{r['gap']*100:>8.1f}p"
        )


# ── 2. 규칙 판별 성능 ───────────────────────────────────────


def measure_rules(rows: Sequence[Any]) -> Mapping[str, Any]:
    out: Dict[str, Any] = {}
    for name, rule in RULES.items():
        tp = fp = fn = tn = 0
        for episode, _models, generations in rows:
            flagged = rule(episode_text(episode))
            target = generations == 4
            if flagged and target:
                tp += 1
            elif flagged:
                fp += 1
            elif target:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "flagged_share": (tp + fp) / len(rows),
        }
    return out


def report_rules(res: Mapping[str, Any]) -> None:
    print("\n[2] 판별 규칙 성능")
    print(f"  {'규칙':<30}{'정밀도':>8}{'재현율':>8}{'F1':>7}{'맞춤':>7}{'오탐':>7}{'놓침':>7}")
    for name, r in res.items():
        print(
            f"  {name:<30}{r['precision']*100:>7.1f}%{r['recall']*100:>7.1f}%"
            f"{r['f1']*100:>6.1f}%{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}"
        )


# ── 3. 집단별 보정의 실제 효과 ──────────────────────────────


def measure_conditional_smearing(
    artifact: Any, rule: Callable[[str], bool]
) -> Mapping[str, Any]:
    """Train에서 집단별 보정계수를 적합해 Dev에서 검증한다.

    비교 대상:
      none        보정 없음 (현재 baseline)
      global      전역 보정계수 하나 (Duan smearing)
      conditional 규칙이 판별한 집단별 보정계수 2개
    """

    def gather(split: str) -> Dict[str, Dict[bool, List[Tuple[float, float]]]]:
        out: Dict[str, Dict[bool, List[Tuple[float, float]]]] = {
            m: {True: [], False: []} for m in MODELS
        }
        for episode, models, _generations in load_split(split):
            flagged = rule(episode_text(episode))
            _scores, costs = hash_regex.predict_episode(episode, artifact)
            for model_id in MODELS:
                out[model_id][flagged].append(
                    (costs[model_id], actual_cost(model_id, models[model_id]))
                )
        return out

    train, dev = gather("train"), gather("dev")
    rows: Dict[str, Any] = {}
    for model_id in MODELS:
        pairs_all = train[model_id][True] + train[model_id][False]
        global_factor = statistics.fmean(
            a / p for p, a in pairs_all if p > 0 and a > 0
        )
        factors: Dict[bool, float] = {}
        for flagged in (True, False):
            pairs = train[model_id][flagged]
            factors[flagged] = (
                statistics.fmean(a / p for p, a in pairs if p > 0 and a > 0)
                if pairs
                else global_factor
            )

        dev_all = dev[model_id][True] + dev[model_id][False]
        p_sum = sum(p for p, _a in dev_all)
        a_sum = sum(a for _p, a in dev_all)
        p_global = p_sum * global_factor
        p_cond = sum(
            p * factors[flagged]
            for flagged in (True, False)
            for p, _a in dev[model_id][flagged]
        )
        rows[model_id] = {
            "global_factor": global_factor,
            "factor_flagged": factors[True],
            "factor_unflagged": factors[False],
            "bias_none": a_sum / p_sum - 1.0,
            "bias_global": a_sum / p_global - 1.0,
            "bias_conditional": a_sum / p_cond - 1.0,
        }
    return rows


def report_conditional_smearing(rows: Mapping[str, Any]) -> None:
    print("\n[3] 집단별 보정의 효과 — Train에서 적합, Dev에서 검증")
    print(
        f"  {'모델':<15}{'보정없음':>10}{'전역보정':>10}{'집단별':>10}"
        f"{'계수(판별)':>11}{'계수(그외)':>11}"
    )
    for model_id in MODELS:
        r = rows[model_id]
        print(
            f"  {LABEL[model_id]:<15}{r['bias_none']*100:>9.1f}%"
            f"{r['bias_global']*100:>9.1f}%{r['bias_conditional']*100:>9.1f}%"
            f"{r['factor_flagged']:>11.3f}{r['factor_unflagged']:>11.3f}"
        )


# ── 4. 꼬리 집중도 ──────────────────────────────────────────


def measure_tail_concentration(rule: Callable[[str], bool]) -> Mapping[str, Any]:
    """실제 비용 총합에서 상위 소수 문항이 차지하는 비중.

    집단별 보정이 듣지 않는 이유를 가른다. 집단 평균의 차이 때문이라면 보정이 듣고,
    집단 안에서 극단값이 총합을 지배한다면 어떤 평균 보정으로도 안정되지 않는다.
    """

    rows = load_split("dev")
    out: Dict[str, Any] = {"models": {}}
    for model_id in MODELS:
        costs = sorted(
            (actual_cost(model_id, models[model_id]) for _e, models, _g in rows),
            reverse=True,
        )
        total = sum(costs)
        entry: Dict[str, Any] = {"total": total}
        for share in (0.01, 0.05, 0.10):
            k = max(1, int(len(costs) * share))
            entry[f"top_{int(share*100)}pct_share"] = sum(costs[:k]) / total
        entry["max_single_share"] = costs[0] / total
        # 판별 집단 안에서의 집중도
        flagged = sorted(
            (
                actual_cost(model_id, models[model_id])
                for e, models, _g in rows
                if rule(episode_text(e))
            ),
            reverse=True,
        )
        if flagged:
            entry["within_flagged_top10pct_share"] = (
                sum(flagged[: max(1, len(flagged) // 10)]) / sum(flagged)
            )
        out["models"][model_id] = entry
    return out


def report_tail_concentration(res: Mapping[str, Any]) -> None:
    print("\n[4] 실제 비용 총합의 꼬리 집중도 (Dev 880)")
    print(
        f"  {'모델':<15}{'상위1%':>9}{'상위5%':>9}{'상위10%':>9}"
        f"{'최대1문항':>10}{'판별군내 상위10%':>17}"
    )
    for model_id in MODELS:
        r = res["models"][model_id]
        print(
            f"  {LABEL[model_id]:<15}{r['top_1pct_share']*100:>8.1f}%"
            f"{r['top_5pct_share']*100:>8.1f}%{r['top_10pct_share']*100:>8.1f}%"
            f"{r['max_single_share']*100:>9.1f}%"
            f"{r.get('within_flagged_top10pct_share', 0)*100:>16.1f}%"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subpopulation",
        description="편향이 집중된 문항군을 프롬프트만으로 식별한다.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "baselines" / "hash-regex-public.v1.json",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = hash_regex.load_artifact(args.artifact)
    dev_rows = load_split("dev")

    print("=" * 74)
    print("편향 집중 문항군 식별  (Dev 880 기준, 보정계수는 Train 1760에서 적합)")
    print("=" * 74)

    signals = measure_signals(dev_rows)
    report_signals(signals)
    rules = measure_rules(dev_rows)
    report_rules(rules)
    smearing = measure_conditional_smearing(artifact, rule_word_problem)
    report_conditional_smearing(smearing)
    tail = measure_tail_concentration(rule_word_problem)
    report_tail_concentration(tail)

    results = {
        "signals": signals,
        "rules": rules,
        "conditional_smearing": smearing,
        "tail_concentration": tail,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.json}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
