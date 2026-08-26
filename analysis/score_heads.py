# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""점수 예측 헤드 — 지금까지 한 번도 건드리지 않은 축.

위험 통제가 끝나 통과율이 100% 가 되자, 남은 손실은 전부 "통과했을 때의 품질"에서
온다. 반칙 진단(`--truth score`)으로 상한을 재보면 점수 예측을 참값으로 갈아끼울 때
0.6729 → 0.7647 (+0.0918) 이다. 안전계수 쪽 튜닝으로 얻은 전부(+0.0246)의 3.7배다.

## 왜 로지스틱인가

`outcomes.json` 의 score 는 [0,1] 연속값이 아니라 `num_generations` 번의 시행 중
성공 비율이다(2 또는 4). 즉 이항 비율이다. 그런데 baseline 은 이걸 그냥 선형 회귀로
맞춘다 — 예측이 [0,1] 밖으로 나갈 수 있고, 분산이 p(1−p)/n 로 p 에 따라 달라지는데
등분산을 가정한다.

로짓 공간에서 맞추면 두 문제가 같이 풀린다. 시행 수를 가중치로 주면 4회 문항이
2회 문항보다 더 신뢰받는다.

## 제출 제약

컨테이너는 표준 라이브러리만 쓴다. 학습은 numpy 로 하되 산출물은 계수 벡터뿐이고,
추론은 내적 한 번 + 시그모이드다 — 순수 파이썬으로 충분하다. 트리 계열을 쓰지 않는
이유가 이것이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.quantile_cost import MODELS, _require_numpy  # noqa: E402

LIGHT_MODEL = "ax31-light"
from baselines import hash_regex  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402


def standardizer(bins: int) -> Tuple[Any, Any]:
    """Train 에서 표준화 계수를 직접 계산한다.

    아티팩트의 feature_mean/scale 은 bins=256 전용이라 차원이 다르면 못 쓴다.
    scale 이 0 인 열(항상 같은 값)은 1 로 둔다 — 나누기 폭발을 막고, 표준화 후
    그 열은 0 이 되어 절편에 흡수된다.
    """

    numpy = _require_numpy()
    episodes = load_input(
        ROOT / "data" / "materialized" / "train" / "inputs.json"
    ).episodes
    raw = numpy.asarray(
        [hash_regex.raw_feature_vector(e, bins) for e in episodes],
        dtype=numpy.float64,
    )
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def build_multi(
    splits: Sequence[str], artifact: Any, bins: int | None = None
) -> Tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """여러 split 을 이어붙인 학습 행렬.

    측정에서는 Dev 를 빼두어야 하지만 **제출하는 아티팩트까지 그럴 이유는 없다** —
    규칙상 공개 Train/Dev 둘 다 학습에 쓸 수 있고(CHALLENGE_RULES.md), 채점셋은
    양쪽과 분리돼 있다. 학습곡선이 1,760 에서 아직 내려가는 중이므로 Dev 880 을
    더하면 예측이 개선된다.
    """

    numpy = _require_numpy()
    mats, scores, trials = [], None, None
    for split in splits:
        matrix, score, trial = build_score_matrix(split, artifact, bins)
        mats.append(matrix)
        if scores is None:
            scores = {m: [score[m]] for m in score}
            trials = {m: [trial[m]] for m in trial}
        else:
            for m in score:
                scores[m].append(score[m])
                trials[m].append(trial[m])
    return (
        numpy.vstack(mats),
        {m: numpy.concatenate(v) for m, v in scores.items()},
        {m: numpy.concatenate(v) for m, v in trials.items()},
    )


def build_score_matrix(
    split: str, artifact: Any, bins: int | None = None
) -> Tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """표준화 특징 행렬 + 모델별 실제 점수 + 시행 수.

    bins 를 주면 그 해상도로 해시 특징을 다시 만들고 표준화도 Train 에서 새로 낸다.
    None 이면 아티팩트와 동일한 특징이라 baseline 재현이 성립한다.
    """

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
    if bins is None or bins == artifact.hash_bins:
        width = artifact.hash_bins
        mean = numpy.asarray(artifact.feature_mean, dtype=numpy.float64)
        scale = numpy.asarray(artifact.feature_scale, dtype=numpy.float64)
    else:
        width = bins
        mean, scale = standardizer(bins)

    rows = []
    scores: Dict[str, List[float]] = {m: [] for m in MODELS}
    trials: Dict[str, List[float]] = {m: [] for m in MODELS}
    for episode in episodes:
        raw = numpy.asarray(
            hash_regex.raw_feature_vector(episode, width),
            dtype=numpy.float64,
        )
        rows.append((raw - mean) / scale)
        record = outcomes[episode.episode_id]
        for model_id in MODELS:
            scores[model_id].append(float(record[model_id]["score"]))
            trials[model_id].append(float(record[model_id]["num_generations"]))
    matrix = numpy.asarray(rows, dtype=numpy.float64)
    # 절편 열 — quantile_cost.build_matrix 와 같은 규약. 이게 없으면 벌점 마스크의
    # 0번 열이 절편이 아니라 실제 특징에 걸린다.
    bias = numpy.ones((matrix.shape[0], 1), dtype=numpy.float64)
    matrix = numpy.hstack([bias, matrix])
    return (
        matrix,
        {m: numpy.asarray(scores[m], dtype=numpy.float64) for m in MODELS},
        {m: numpy.asarray(trials[m], dtype=numpy.float64) for m in MODELS},
    )


def fit_ridge(matrix: Any, target: Any, *, l2: float) -> Any:
    """baseline 과 같은 ridge. alpha 를 넓게 쓸 수 있게 노출만 한다."""

    numpy = _require_numpy()
    rows, columns = matrix.shape
    penalty = numpy.eye(columns) * l2
    penalty[0, 0] = 0.0
    return numpy.linalg.solve(
        matrix.T @ matrix / rows + penalty, matrix.T @ target / rows
    )


def fit_binomial(
    matrix: Any,
    successes: Any,
    trials: Any,
    *,
    l2: float,
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> Any:
    """이항 로지스틱 회귀 (IRLS). score 는 시행 중 성공 비율이므로 이게 맞는 모형이다.

    시행 수를 가중치로 준다 — 4회 문항의 관측이 2회 문항보다 정보량이 크다.
    절편에는 벌점을 걸지 않는다(baseline ridge 규약과 동일).
    """

    numpy = _require_numpy()
    rows, columns = matrix.shape
    weights = numpy.zeros(columns)
    penalty = numpy.eye(columns) * l2
    penalty[0, 0] = 0.0

    for _ in range(iterations):
        logits = numpy.clip(matrix @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + numpy.exp(-logits))
        # IRLS 가중치. 시행 수를 곱해 관측 신뢰도를 반영한다.
        variance = numpy.maximum(probability * (1.0 - probability), 1e-6) * trials
        working = logits + (successes - probability) * trials / variance
        left = (matrix.T * variance) @ matrix / rows + penalty
        right = (matrix.T * variance) @ working / rows
        updated = numpy.linalg.solve(left, right)
        if numpy.max(numpy.abs(updated - weights)) < tolerance:
            weights = updated
            break
        weights = updated
    return weights


def predict_binomial(matrix: Any, weights: Any) -> Any:
    numpy = _require_numpy()
    return 1.0 / (1.0 + numpy.exp(-numpy.clip(matrix @ weights, -30.0, 30.0)))


def build_score_heads(
    artifact: Any,
    *,
    ridge_l2: Tuple[float, ...],
    binomial_l2: Tuple[float, ...],
    bins: int | None = None,
    light_l2: Tuple[float, ...] = (),
    default_l2: float = 1000.0 / 1760.0,
    boost: Tuple[float, ...] = (),
    shrink: Tuple[Tuple[float, ...], ...] = (),
    sparse_uplift: Tuple[Tuple[str, ...], ...] = (),
    interact: Tuple[float, ...] = (),
    shrink_train: Tuple[Tuple[float, ...], ...] = (),
) -> Mapping[str, Mapping[str, Any]]:
    """Train 에서 점수 헤드들을 적합해 Dev 예측을 만든다.

    반환: 헤드 이름 -> {model_id: Dev 문항별 점수 예측}
    """

    numpy = _require_numpy()
    xtr, ytr, ttr = build_score_matrix("train", artifact, bins)
    xdv, _ydv, _tdv = build_score_matrix("dev", artifact, bins)
    out: Dict[str, Dict[str, Any]] = {}

    for model_id in MODELS:
        for value in ridge_l2:
            weights = fit_ridge(xtr, ytr[model_id], l2=value)
            out.setdefault(f"score_ridge{value:g}", {})[model_id] = numpy.clip(
                xdv @ weights, 0.0, 1.0
            )
        for value in light_l2:
            # light 전용 alpha — light 는 모든 문항의 기본 선택이라 승급 판단
            # (승급 점수 − light 점수)의 기준점이 된다. 참값 진단상 light 하나만
            # 고쳐도 전체 점수 이득의 절반이 나오므로 별도로 조율할 값이 있다.
            base = f"score_light{value:g}"
            if model_id == LIGHT_MODEL:
                fitted = fit_ridge(xtr, ytr[model_id], l2=value)
                out.setdefault(base, {})[model_id] = numpy.clip(
                    xdv @ fitted, 0.0, 1.0
                )
            else:
                fitted = fit_ridge(xtr, ytr[model_id], l2=default_l2)
                out.setdefault(base, {})[model_id] = numpy.clip(
                    xdv @ fitted, 0.0, 1.0
                )
        for value in binomial_l2:
            weights = fit_binomial(
                xtr, ytr[model_id], ttr[model_id], l2=value
            )
            out.setdefault(f"score_binomial{value:g}", {})[model_id] = (
                predict_binomial(xdv, weights)
            )
    if interact:
        # dense 특징 간 곱항 추가 — 지금까지 선형만 봤다.
        #
        # 업리프트는 "이 문항에서 어느 모델이 상대적으로 나은가"인데, 그건 특징
        # 하나가 아니라 조합으로 결정될 수 있다(예: 길다 AND 코드다). 선형 모델은
        # 그런 조건부 구조를 표현하지 못한다.
        #
        # dense 14개의 모든 쌍은 91개다. 해시버킷은 이미 희소해 곱항을 만들지 않는다.
        import itertools as _it
        import json as _json

        names = _json.loads(
            (ROOT / "baselines" / "hash-regex-public.v1.json").read_text("utf-8")
        )["dense_feature_names"]
        dense = len(names)
        pairs = list(_it.combinations(range(dense), 2))
        cross_tr = numpy.column_stack(
            [xtr[:, 1 + a] * xtr[:, 1 + b] for a, b in pairs]
        )
        cross_dv = numpy.column_stack(
            [xdv[:, 1 + a] * xdv[:, 1 + b] for a, b in pairs]
        )
        wide_tr = numpy.hstack([xtr, cross_tr])
        wide_dv = numpy.hstack([xdv, cross_dv])
        for value in interact:
            for model_id in MODELS:
                weights = fit_ridge(wide_tr, ytr[model_id], l2=value / wide_tr.shape[0])
                out.setdefault(f"score_interact{value:g}", {})[model_id] = numpy.clip(
                    wide_dv @ weights, 0.0, 1.0
                )

    if sparse_uplift:
        # 수준은 전체 특징, 업리프트는 희소 특징.
        #
        # light→ax31 업리프트의 순위상관이 전체 270차원으로는 0.052 인데 dense 특징
        # 3개만 쓰면 0.294 다(상한 0.908). dense 특징들이 강하게 공선적이라 —
        # 전부 길이·복잡도 대리변수다 — 릿지가 Train 잡음에 맞춰 가중치를 나눠 갖고
        # Dev 에서 상쇄된다. 특징을 더할수록 신호가 파괴되는 것이다.
        #
        # 배분기는 문항 간 순위를 쓰므로 이 지표가 곧 목적이다. RMSE 와 달리 대리가
        # 아니다(RMSE 는 세 번 배신했다).
        #
        # think 업리프트는 반대로 전체 특징이 최선(0.290)이라 건드리지 않는다.
        import json as _json

        names = _json.loads(
            (ROOT / "baselines" / "hash-regex-public.v1.json").read_text("utf-8")
        )["dense_feature_names"]
        for spec in sparse_uplift:
            picked = [1 + names.index(n) for n in spec]
            columns = [0] + picked
            level = {}
            for model_id in MODELS:
                weights = fit_ridge(xtr, ytr[model_id], l2=default_l2)
                level[model_id] = numpy.clip(xdv @ weights, 0.0, 1.0)
            uplift_w = fit_ridge(
                xtr[:, columns],
                ytr["ax31"] - ytr[LIGHT_MODEL],
                l2=1.0 / xtr.shape[0],
            )
            key = "score_sparse" + "|".join(spec)
            out.setdefault(key, {})[LIGHT_MODEL] = level[LIGHT_MODEL]
            out[key]["axk1-think"] = level["axk1-think"]
            out[key]["ax31"] = numpy.clip(
                level[LIGHT_MODEL] + xdv[:, columns] @ uplift_w, 0.0, 1.0
            )

    if shrink_train:
        # 배포 가능한 수축 — 중심을 **Train 예측 평균**으로 잡고 클립을 마지막에 한 번만.
        #
        # shrink 변형은 Dev 예측 평균을 중심으로 썼는데 그것은 아티팩트에 구울 수 없다
        # (배포 시점에 채점셋 평균을 알 수 없다). 계수로 접으면
        #
        #     p' = m + g(b + x·w - m) = [g·b + (1-g)·m] + g·(x·w)
        #
        # 즉 절편 g·b + (1-g)·m, 계수 g·w 다. 여기서 재는 것이 곧 구워지는 것이다.
        for spec in shrink_train:
            gammas = dict(zip(MODELS, spec))
            key = "score_shrinkt" + ":".join(f"{g:g}" for g in spec)
            for model_id in MODELS:
                weights = fit_ridge(xtr, ytr[model_id], l2=default_l2)
                centre = float((xtr @ weights).mean())
                gamma = gammas[model_id]
                folded = weights * gamma
                folded[0] = gamma * weights[0] + (1.0 - gamma) * centre
                out.setdefault(key, {})[model_id] = numpy.clip(
                    xdv @ folded, 0.0, 1.0
                )

    if shrink:
        # 예측 분산을 모델별로 누른다: p' = mean + gamma*(p - mean)
        #
        # 배분기는 점수의 절대값이 아니라 문항 간 순위를 쓴다. 그런데 light→ax31
        # 업리프트의 순위상관이 0.04 로 사실상 0 이다 — 그 순위를 믿는 것은 잡음을
        # 믿는 것이다. gamma=0 이면 그 모델은 전 문항 동일 점수가 되어 배분기가
        # 비용 효율로만 승급을 고른다("어느 것"이 아니라 "몇 개").
        #
        # alpha 를 키우는 것과 방향은 같지만 두 가지가 다르다: 모델별로 따로 걸 수
        # 있고, 적합을 바꾸지 않아 순위 정보의 유무만 분리해서 볼 수 있다.
        for spec in shrink:
            gammas = dict(zip(MODELS, spec))
            for model_id in MODELS:
                weights = fit_ridge(xtr, ytr[model_id], l2=default_l2)
                raw_pred = numpy.clip(xdv @ weights, 0.0, 1.0)
                gamma = gammas[model_id]
                centre = float(raw_pred.mean())
                key = "score_shrink" + ":".join(f"{g:g}" for g in spec)
                out.setdefault(key, {})[model_id] = numpy.clip(
                    centre + gamma * (raw_pred - centre), 0.0, 1.0
                )

    if boost:
        from analysis.boosted_score import fit_boosted, predict

        # blend 는 ridge 비중. 0 이면 GBM 단독, 0.5 면 50:50.
        # 트리 설정은 고정한다 — 여기서 또 스윕하면 비교 후보만 늘어난다.
        cache = {}
        for model_id in MODELS:
            model = fit_boosted(
                xtr, ytr[model_id],
                trees=60, depth=3, learning_rate=0.08, bins=24, min_leaf=25,
            )
            cache[model_id] = numpy.clip(predict(xdv, model), 0.0, 1.0)
        for blend in boost:
            for model_id in MODELS:
                weights = fit_ridge(xtr, ytr[model_id], l2=default_l2)
                linear = numpy.clip(xdv @ weights, 0.0, 1.0)
                out.setdefault(f"score_boost{blend:g}", {})[model_id] = numpy.clip(
                    blend * linear + (1.0 - blend) * cache[model_id], 0.0, 1.0
                )
    return out
