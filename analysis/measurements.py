# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""보고서에 실린 모든 수치를 한 번에 재생성한다.

`research-notes.md`와 공유 문서의 숫자는 전부 이 스크립트가 출력한 값이다.
코드나 artifact가 바뀌면 다시 돌려 숫자를 갱신한다.

사용:
    python3 analysis/measurements.py                       # 전체
    python3 analysis/measurements.py --only bias smearing  # 일부
    python3 analysis/measurements.py --json build/m.json   # 기계 판독용

전제: data/materialized/{train,dev}/inputs.json 이 생성돼 있어야 한다
      (tools/materialize_public_data.py). 표준 라이브러리만 사용한다.
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
from ossp_router.heuristic import episode_text, extract_features  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    Episode,
    load_bundled_policy,
    load_input,
)

MODELS = ("ax31-light", "ax31", "axk1-think")
LABEL = {"ax31-light": "싼 모델", "ax31": "중간 모델", "axk1-think": "비싼 추론 모델"}

# docs/SCORING.md 의 확정 계수. 정책 파일과 이중으로 검증한다.
RATES = {
    "ax31-light": (Decimal("1"), Decimal("4")),
    "ax31": (Decimal("2.127"), Decimal("8.509")),
    "axk1-think": (Decimal("6.565"), Decimal("26.260")),
}
TOKEN_UNIT = Decimal("1000000")


# ── 공통 로딩 ────────────────────────────────────────────────


def _artifact(path: Path) -> Any:
    return hash_regex.load_artifact(path)


def _outcomes(split: str) -> Mapping[str, Any]:
    raw = json.loads((ROOT / "data" / split / "outcomes.json").read_text("utf-8"))
    return {row["episode_id"]: row for row in raw["episodes"]}


def _episodes(split: str) -> Sequence[Episode]:
    return load_input(ROOT / "data" / "materialized" / split / "inputs.json").episodes


def _actual_cost(model_id: str, entry: Mapping[str, Any]) -> float:
    rate_in, rate_out = RATES[model_id]
    total = (
        Decimal(entry["input_tokens"]) * rate_in
        + Decimal(entry["output_tokens"]) * rate_out
    ) / TOKEN_UNIT
    return float(total)


def _predict(split: str, artifact: Any) -> Tuple[Sequence[Episode], List[Any]]:
    episodes = _episodes(split)
    return episodes, [hash_regex.predict_episode(e, artifact) for e in episodes]


def _check_rates(policy: Any) -> None:
    """RATES 상수가 번들 정책과 일치하는지 확인 — 어긋나면 모든 수치가 무의미해진다."""

    for model_id, (rate_in, rate_out) in RATES.items():
        entry = policy.models[model_id]
        if (
            Decimal(str(entry.input_token_rate)) != rate_in
            or Decimal(str(entry.output_token_rate)) != rate_out
        ):
            raise SystemExit(
                f"계수 불일치: {model_id} 정책={entry.input_token_rate}/"
                f"{entry.output_token_rate} 스크립트={rate_in}/{rate_out}"
            )


# ── 1. 비용 편향과 Jensen 성분 ──────────────────────────────


def measure_bias(artifact: Any) -> Mapping[str, Any]:
    """예측 총합 대비 실제 총합의 편향. log→exp 되돌림이 원인인지 함께 본다.

    로그 공간에서 회귀하고 exp를 취하면 기하평균에 가까운 값이 나온다.
    잔차가 대략 로그정규라면 산술평균과의 비는 exp(sigma^2/2) 이므로,
    그 값이 관측 편향을 어느 정도 설명하는지 비교한다.
    """

    episodes, preds = _predict("dev", artifact)
    outcomes = _outcomes("dev")
    rows: Dict[str, Any] = {}
    for model_id in MODELS:
        pred_sum = act_sum = 0.0
        residuals: List[float] = []
        for episode, (_scores, costs) in zip(episodes, preds):
            predicted = costs[model_id]
            actual = _actual_cost(
                model_id, outcomes[episode.episode_id]["models"][model_id]
            )
            pred_sum += predicted
            act_sum += actual
            if predicted > 0 and actual > 0:
                residuals.append(math.log(actual) - math.log(predicted))
        sd = statistics.pstdev(residuals)
        bias = act_sum / pred_sum - 1.0
        jensen = math.exp(sd * sd / 2) - 1.0
        rows[model_id] = {
            "predicted_sum": pred_sum,
            "actual_sum": act_sum,
            "underestimate": bias,
            "log_residual_sd": sd,
            "jensen_predicted_bias": jensen,
            "jensen_share_of_bias": jensen / bias if bias else None,
        }
    return rows


def report_bias(rows: Mapping[str, Any]) -> None:
    print("\n[1] 비용 총합 편향과 log→exp 되돌림 성분")
    print(
        f"  {'모델':<14}{'예측합':>10}{'실제합':>10}{'과소추정':>10}"
        f"{'로그잔차σ':>11}{'exp(σ²/2)':>11}{'설명력':>8}"
    )
    for model_id in MODELS:
        r = rows[model_id]
        share = r["jensen_share_of_bias"]
        print(
            f"  {LABEL[model_id]:<14}{r['predicted_sum']:>10.4f}{r['actual_sum']:>10.4f}"
            f"{r['underestimate']*100:>9.1f}%{r['log_residual_sd']:>11.3f}"
            f"{r['jensen_predicted_bias']*100:>10.1f}%"
            f"{(share*100 if share else 0):>7.0f}%"
        )


# ── 2. Duan smearing 보정 ───────────────────────────────────


def measure_smearing(artifact: Any) -> Mapping[str, Any]:
    """Train에서 보정계수를 구해 Dev에서 검증한다 (Duan 1983).

    보정계수 = mean(actual / predicted). Dev로 구해 Dev에서 확인하면 무의미하므로
    반드시 Train에서 적합한다.
    """

    def sums(split: str) -> Mapping[str, Tuple[List[float], List[float]]]:
        episodes, preds = _predict(split, artifact)
        outcomes = _outcomes(split)
        out: Dict[str, Tuple[List[float], List[float]]] = {
            m: ([], []) for m in MODELS
        }
        for episode, (_scores, costs) in zip(episodes, preds):
            for model_id in MODELS:
                out[model_id][0].append(costs[model_id])
                out[model_id][1].append(
                    _actual_cost(
                        model_id, outcomes[episode.episode_id]["models"][model_id]
                    )
                )
        return out

    train, dev = sums("train"), sums("dev")
    rows: Dict[str, Any] = {}
    for model_id in MODELS:
        pred_t, act_t = train[model_id]
        ratios = [a / p for p, a in zip(pred_t, act_t) if p > 0 and a > 0]
        factor = statistics.fmean(ratios)
        pred_d, act_d = dev[model_id]
        p_sum, a_sum = sum(pred_d), sum(act_d)
        rows[model_id] = {
            "smearing_factor_from_train": factor,
            "dev_bias_before": a_sum / p_sum - 1.0,
            "dev_bias_after": a_sum / (p_sum * factor) - 1.0,
        }
    return rows


def report_smearing(rows: Mapping[str, Any]) -> None:
    print("\n[2] Duan smearing — Train에서 계수 적합, Dev에서 검증")
    print(f"  {'모델':<14}{'보정계수':>10}{'보정 전':>10}{'보정 후':>10}")
    for model_id in MODELS:
        r = rows[model_id]
        print(
            f"  {LABEL[model_id]:<14}{r['smearing_factor_from_train']:>10.3f}"
            f"{r['dev_bias_before']*100:>9.1f}%{r['dev_bias_after']*100:>9.1f}%"
        )


# ── 3. 낯선 어휘 비율에 따른 편향 ───────────────────────────


def measure_oov(artifact: Any) -> Mapping[str, Any]:
    """Train 어휘 기준 미등장 토큰 비율로 Dev를 3등분해 편향을 비교한다.

    주의: 미등장 비율과 문항 종류가 얽혀 있어 인과를 분리하지 못한다.
    """

    vocab = set()
    for episode in _episodes("train"):
        vocab.update(hash_regex._normalized_tokens(episode_text(episode)))

    episodes, preds = _predict("dev", artifact)
    outcomes = _outcomes("dev")
    rows: List[Tuple[float, Mapping[str, Tuple[float, float]]]] = []
    for episode, (_scores, costs) in zip(episodes, preds):
        tokens = hash_regex._normalized_tokens(episode_text(episode))
        if not tokens:
            continue
        oov = sum(1 for t in tokens if t not in vocab) / len(tokens)
        rows.append(
            (
                oov,
                {
                    m: (
                        costs[m],
                        _actual_cost(
                            m, outcomes[episode.episode_id]["models"][m]
                        ),
                    )
                    for m in MODELS
                },
            )
        )

    ratios = [r[0] for r in rows]
    rows.sort(key=lambda r: r[0])
    third = len(rows) // 3
    groups = (
        ("낮음(하위 1/3)", rows[:third]),
        ("중간", rows[third : 2 * third]),
        ("높음(상위 1/3)", rows[2 * third :]),
    )
    result: Dict[str, Any] = {
        "vocabulary_size_train": len(vocab),
        "hash_bins": artifact.hash_bins,
        "words_per_bin": len(vocab) / artifact.hash_bins,
        "dev_oov_median": statistics.median(ratios),
        "dev_oov_mean": statistics.fmean(ratios),
        "dev_oov_max": max(ratios),
        "dev_episodes_over_half_oov": sum(1 for x in ratios if x > 0.5),
        "dev_episodes": len(rows),
        "tertiles": [],
    }
    for name, group in groups:
        entry = {
            "group": name,
            "oov_mean": statistics.fmean(x[0] for x in group),
            "bias": {},
        }
        for model_id in MODELS:
            p = sum(x[1][model_id][0] for x in group)
            a = sum(x[1][model_id][1] for x in group)
            entry["bias"][model_id] = a / p - 1.0
        result["tertiles"].append(entry)
    return result


def report_oov(res: Mapping[str, Any]) -> None:
    print("\n[3] 낯선 어휘 비율과 비용 편향")
    print(
        f"  Train 서로 다른 토큰 {res['vocabulary_size_train']:,}개 / "
        f"저장 칸 {res['hash_bins']}개 → 칸당 평균 {res['words_per_bin']:,.0f}개"
    )
    print(
        f"  Dev 미등장 비율: 중앙 {res['dev_oov_median']*100:.1f}% · "
        f"평균 {res['dev_oov_mean']*100:.1f}% · 최대 {res['dev_oov_max']*100:.1f}% · "
        f"절반 초과 문항 {res['dev_episodes_over_half_oov']}/{res['dev_episodes']}"
    )
    print(f"  {'구간':<18}{'미등장':>9}{'싼':>9}{'중간':>9}{'비싼':>9}")
    for t in res["tertiles"]:
        cells = "".join(f"{t['bias'][m]*100:>8.1f}%" for m in MODELS)
        print(f"  {t['group']:<18}{t['oov_mean']*100:>8.1f}%{cells}")


# ── 4. 점수 예측 품질 ───────────────────────────────────────


def measure_score_quality(artifact: Any) -> Mapping[str, Any]:
    """승격이 실제로 점수를 올리는 문항을 예측이 골라내는지 본다."""

    episodes, preds = _predict("dev", artifact)
    outcomes = _outcomes("dev")
    pred: Dict[str, List[float]] = {m: [] for m in MODELS}
    act: Dict[str, List[float]] = {m: [] for m in MODELS}
    for episode, (scores, _costs) in zip(episodes, preds):
        for model_id in MODELS:
            pred[model_id].append(scores[model_id])
            act[model_id].append(
                float(
                    Decimal(
                        outcomes[episode.episode_id]["models"][model_id]["score"]
                    )
                )
            )

    light, think = MODELS[0], MODELS[2]
    gain_act = [a - b for b, a in zip(act[light], act[think])]
    gain_pred = [a - b for b, a in zip(pred[light], pred[think])]
    helps = [i for i, v in enumerate(gain_act) if v > 0]
    hurts = [i for i, v in enumerate(gain_act) if v < 0]
    same = [i for i, v in enumerate(gain_act) if v == 0]

    order = sorted(range(len(gain_pred)), key=lambda i: -gain_pred[i])
    top: Dict[str, Any] = {}
    for n in (50, 100, 200):
        hit = sum(1 for i in order[:n] if gain_act[i] > 0)
        top[str(n)] = {"hit": hit, "n": n, "precision": hit / n}

    return {
        "per_model": {
            m: {
                "predicted_mean": statistics.fmean(pred[m]),
                "actual_mean": statistics.fmean(act[m]),
                "mae": statistics.fmean(
                    abs(p - a) for p, a in zip(pred[m], act[m])
                ),
            }
            for m in MODELS
        },
        "upgrade_effect": {
            "helps": len(helps),
            "hurts": len(hurts),
            "no_change": len(same),
            "predicted_gain_on_helps": statistics.fmean(gain_pred[i] for i in helps),
            "predicted_gain_on_hurts": statistics.fmean(gain_pred[i] for i in hurts),
            "predicted_gain_on_no_change": statistics.fmean(
                gain_pred[i] for i in same
            ),
        },
        "top_n_precision": top,
        "base_rate": len(helps) / len(gain_act),
    }


def report_score_quality(res: Mapping[str, Any]) -> None:
    print("\n[4] 점수 예측 품질")
    print(f"  {'모델':<14}{'예측평균':>10}{'실제평균':>10}{'MAE':>8}")
    for model_id in MODELS:
        r = res["per_model"][model_id]
        print(
            f"  {LABEL[model_id]:<14}{r['predicted_mean']:>10.3f}"
            f"{r['actual_mean']:>10.3f}{r['mae']:>8.3f}"
        )
    u = res["upgrade_effect"]
    print(
        f"  비싼 모델 승격 효과: 이득 {u['helps']} · 손해 {u['hurts']} · "
        f"무차이 {u['no_change']}"
    )
    print(
        f"    예측 이득폭 평균 — 이득 {u['predicted_gain_on_helps']:+.4f} · "
        f"손해 {u['predicted_gain_on_hurts']:+.4f} · "
        f"무차이 {u['predicted_gain_on_no_change']:+.4f}  ← 손해와 무차이를 구분 못 함"
    )
    for key, t in res["top_n_precision"].items():
        print(
            f"    예측 상위 {key:>3}개 중 실제 이득 {t['hit']}/{t['n']} "
            f"({t['precision']*100:.0f}%)   무작위 기대 {res['base_rate']*100:.0f}%"
        )


# ── 5. 계수 무게 분포 ───────────────────────────────────────


def measure_coefficient_mass(artifact_path: Path) -> Mapping[str, Any]:
    """명시 특징과 해시 버킷 중 어디에 계수 크기가 실렸는지.

    지표일 뿐 인과가 아니다. 표준화된 특징이므로 |계수| 합은 대략적인 영향력 대리값.
    """

    raw = json.loads(artifact_path.read_text("utf-8"))
    dense_n = len(raw["dense_feature_names"])
    rows: Dict[str, Any] = {"dense_feature_count": dense_n, "heads": {}}
    for kind in ("score_heads", "log_cost_heads"):
        for model_id, head in sorted(raw[kind].items()):
            coefficients = head["coefficients"]
            dense = sum(abs(x) for x in coefficients[:dense_n])
            hashed = sum(abs(x) for x in coefficients[dense_n:])
            rows["heads"][f"{kind}:{model_id}"] = {
                "dense_abs_sum": dense,
                "hashed_abs_sum": hashed,
                "dense_share": dense / (dense + hashed),
            }
    return rows


def report_coefficient_mass(res: Mapping[str, Any]) -> None:
    print("\n[5] 계수 크기 비중 — 명시 특징 vs 해시 버킷")
    print(f"  {'head':<30}{'명시':>9}{'해시':>9}{'명시비중':>10}")
    for name, r in res["heads"].items():
        print(
            f"  {name:<30}{r['dense_abs_sum']:>9.2f}{r['hashed_abs_sum']:>9.2f}"
            f"{r['dense_share']*100:>9.0f}%"
        )


# ── 6. 표면 표기 민감도 ─────────────────────────────────────

SENSITIVITY_PROMPTS = (
    ("원문", "이 명제를 증명하시오."),
    ("동의어", "이 명제를 입증하시오."),
    ("어미 변경", "이 명제를 증명하세요."),
    ("무의미 단어", "끄루뽁 즈믈캬 뾰롱뾰롱 흐뭅뜨 꺄륵꺄륵 뿌긔뿌긔 삐룡삐룡"),
)


def measure_surface_sensitivity(artifact: Any) -> Mapping[str, Any]:
    """뜻이 같아도 표기가 다르면 예측이 달라지는 정도.

    마지막 항목은 Train에 없을 단어들 — 사전 조회가 없으므로 예측이 그래도 나온다는 확인.
    """

    rows = []
    for label, text in SENSITIVITY_PROMPTS:
        scores, costs = hash_regex.predict_episode(
            Episode(episode_id="probe", prompt=text), artifact
        )
        rows.append(
            {
                "label": label,
                "prompt": text,
                "tokens": list(hash_regex._normalized_tokens(text)),
                "score_light": scores[MODELS[0]],
                "score_think": scores[MODELS[2]],
                "cost_think_per_million": costs[MODELS[2]] * 1_000_000,
            }
        )
    base = rows[0]["cost_think_per_million"]
    for row in rows:
        row["cost_vs_base"] = row["cost_think_per_million"] / base - 1.0
    return {"probes": rows}


def report_surface_sensitivity(res: Mapping[str, Any]) -> None:
    print("\n[6] 표면 표기 민감도 (비싼 모델 비용 예측)")
    print(f"  {'구분':<14}{'비용예측':>12}{'원문 대비':>11}  프롬프트")
    for row in res["probes"]:
        print(
            f"  {row['label']:<14}{row['cost_think_per_million']:>12,.0f}"
            f"{row['cost_vs_base']*100:>10.1f}%  {row['prompt'][:30]}"
        )
    print(f"  토큰화 예: {res['probes'][0]['tokens']}")


# ── 7. 등급별 실질 여유 ─────────────────────────────────────


def measure_headroom(artifact: Any) -> Mapping[str, Any]:
    """안전계수가 남긴 여유와 추정 편향의 크기를 나란히 본다."""

    from analysis.holdout_protocol import apply_policy, score_tier  # 지연 import

    episodes = _episodes("dev")
    inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    outcomes_batch = __import__(
        "ossp_router.protocol", fromlist=["load_outcomes"]
    ).load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    policy = load_bundled_policy()
    preds = [hash_regex.predict_episode(e, artifact) for e in episodes]

    rows: Dict[str, Any] = {}
    for tier in TIERS:
        multiplier = float(policy.tiers[tier].budget_multiplier)
        safety = artifact.tier_safety_ratios[tier]
        selected = apply_policy(preds, policy, tier, safety)
        report = score_tier(inputs, outcomes_batch, policy, tier, selected)
        actual = float(Decimal(report["budget_ratio"]))
        target = multiplier * safety
        rows[tier] = {
            "budget_multiplier": multiplier,
            "safety_ratio": safety,
            "predicted_target": target,
            "actual_budget_ratio": actual,
            "underestimate_vs_target": actual / target - 1.0,
            "margin_left_by_safety": 1.0 - safety,
            "effective_headroom": 1.0 - actual / multiplier,
            "limit_consumed": actual / multiplier,
            "near_budget": report["near_budget"],
            "budget_passed": report["budget_passed"],
        }
    return rows


def report_headroom(rows: Mapping[str, Any]) -> None:
    print("\n[7] 등급별 실질 여유 (공개 안전계수 적용)")
    print(
        f"  {'등급':<10}{'한도':>6}{'안전계수':>9}{'목표':>8}{'실제':>8}"
        f"{'과소':>8}{'남긴여유':>9}{'실질여유':>9}{'소진율':>8}"
    )
    for tier in TIERS:
        r = rows[tier]
        print(
            f"  {tier:<10}{r['budget_multiplier']:>6}{r['safety_ratio']:>9.4f}"
            f"{r['predicted_target']:>8.3f}{r['actual_budget_ratio']:>8.3f}"
            f"{r['underestimate_vs_target']*100:>7.1f}%"
            f"{r['margin_left_by_safety']*100:>8.1f}%"
            f"{r['effective_headroom']*100:>8.1f}%"
            f"{r['limit_consumed']*100:>7.1f}%"
        )


# ── 8. 홀드아웃 리포트에서 안전 목표 도출 ───────────────────


def measure_safe_targets(report_path: Path) -> Mapping[str, Any]:
    """홀드아웃 리포트의 최악 드리프트로부터 보정 목표 상한을 계산한다.

    안전 목표 = 한도 / 최악배율. 보정셋에서 이 값까지만 쓰면
    관측된 최악 시나리오에서도 한도를 넘지 않는다.
    """

    if not report_path.exists():
        return {"error": f"리포트 없음: {report_path}. holdout_protocol.py 먼저 실행"}
    report = json.loads(report_path.read_text("utf-8"))
    strategies = report.get("strategies")
    if strategies is None:
        return {"error": "구버전 리포트(v1). --report 를 v2로 지정"}

    rows: Dict[str, Any] = {}
    for tier in TIERS:
        worst = 0.0
        worst_name = ""
        worst_holdout = 0.0
        limit = None
        trials_seen = 0
        for name, block in strategies.items():
            entry = block["tiers"][tier]
            limit = entry["budget_multiplier"]
            hi = entry["holdout_budget_ratio"]["max"]
            worst_holdout = max(worst_holdout, hi)
            for trial in entry.get("trials", ()):
                trials_seen += 1
                if trial["calib_budget_ratio"] > 0:
                    mult = (
                        trial["holdout_budget_ratio"] / trial["calib_budget_ratio"]
                    )
                    if mult > worst:
                        worst, worst_name = mult, name
        if not trials_seen:
            return {
                "error": "리포트에 trials가 없어 최악 배율을 계산할 수 없다. "
                "trials를 보존한 리포트를 지정할 것"
            }
        rows[tier] = {
            "budget_multiplier": limit,
            "worst_drift_multiplier": worst,
            "worst_strategy": worst_name,
            "worst_holdout_ratio": worst_holdout,
            "worst_holdout_vs_limit": worst_holdout / limit if limit else None,
            "safe_calibration_target": limit / worst if worst else None,
            "safe_target_as_limit_share": (limit / worst) / limit if worst else None,
        }
    return rows


def report_safe_targets(rows: Mapping[str, Any]) -> None:
    print("\n[8] 홀드아웃 최악값에서 도출한 안전 보정 목표")
    if "error" in rows:
        print(f"  건너뜀 — {rows['error']}")
        return
    print(
        f"  {'등급':<10}{'한도':>6}{'최악배율':>10}{'최악전략':<22}"
        f"{'최악/한도':>10}{'안전목표':>10}{'소진율':>8}"
    )
    for tier in TIERS:
        r = rows[tier]
        print(
            f"  {tier:<10}{r['budget_multiplier']:>6}"
            f"{r['worst_drift_multiplier']:>10.3f}{r['worst_strategy']:<22}"
            f"{r['worst_holdout_vs_limit']*100:>9.1f}%"
            f"{r['safe_calibration_target']:>10.3f}"
            f"{r['safe_target_as_limit_share']*100:>7.1f}%"
        )


# ── 실행 ────────────────────────────────────────────────────

SECTIONS = {
    "bias": (measure_bias, report_bias, "artifact"),
    "smearing": (measure_smearing, report_smearing, "artifact"),
    "oov": (measure_oov, report_oov, "artifact"),
    "score": (measure_score_quality, report_score_quality, "artifact"),
    "coefficients": (measure_coefficient_mass, report_coefficient_mass, "path"),
    "surface": (measure_surface_sensitivity, report_surface_sensitivity, "artifact"),
    "headroom": (measure_headroom, report_headroom, "artifact"),
    "safe-targets": (measure_safe_targets, report_safe_targets, "holdout"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measurements",
        description="보고서 수치를 재생성합니다.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "baselines" / "hash-regex-public.v1.json",
    )
    parser.add_argument(
        "--holdout-report",
        type=Path,
        default=ROOT / "analysis" / "reports" / "holdout-gap-v2.json",
    )
    parser.add_argument("--json", type=Path, help="결과를 JSON으로도 저장")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(SECTIONS),
        help="일부 항목만 실행 (기본: 전체)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = load_bundled_policy()
    _check_rates(policy)
    artifact = _artifact(args.artifact)

    names = args.only or list(SECTIONS)
    print("=" * 74)
    print("보고서 수치 재생성")
    print(f"  artifact       {args.artifact.relative_to(ROOT)}")
    print(f"  안전계수       {dict(sorted(artifact.tier_safety_ratios.items()))}")
    print(f"  저장 칸        {artifact.hash_bins}")
    print(f"  항목           {', '.join(names)}")
    print("=" * 74)

    results: Dict[str, Any] = {}
    for name in names:
        measure, report, needs = SECTIONS[name]
        if needs == "artifact":
            value = measure(artifact)
        elif needs == "path":
            value = measure(args.artifact)
        else:
            value = measure(args.holdout_report)
        results[name] = value
        report(value)

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
