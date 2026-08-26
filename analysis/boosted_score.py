# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""부스팅 점수 헤드 — 선형 모델이 못 잡는 몫이 남아 있는가.

점수 참값을 주입하면 +0.0918 인데 ridge 튜닝으로 캔 건 +0.0079 뿐이다. 남은 대부분이
비선형성이거나 특징 상호작용일 수 있다. 독립 근거도 있다 — 다른 참가팀(thislifehea)의
채택본이 ridge+GBM 앙상블이었다.

## 왜 직접 구현하는가

이 저장소의 학습 의존성은 numpy 하나다(`baselines/requirements-train.txt`). sklearn 을
들이는 대신 깊이 3 회귀 트리를 직접 쓴다. 어차피 **컨테이너용으로 트리를 JSON 으로
내보내고 순수 파이썬으로 순회**해야 하므로, 구조를 우리가 소유하는 편이 낫다.

## 구조

    깊이 D 회귀 트리, 제곱손실, 축 정렬 분할
    각 분할은 특징 하나의 임계값 — 추론은 비교 D 번
    학습률 × 트리 M 개를 더한 것이 예측 (초기값은 목표 평균)

분할 후보는 특징별 분위수로 제한한다(`--bins`). 270 차원 × 2,640 행을 전수 탐색하면
느리고, 분위수만 봐도 트리 성능은 거의 같다.

## 검증

`verify_pure_python()` 이 numpy 예측과 순수 파이썬 순회 결과를 대조한다. 컨테이너에서
도는 것은 후자이므로 이 대조가 통과해야 배포할 수 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.quantile_cost import _require_numpy  # noqa: E402

# 리프 노드는 (None, 값), 내부 노드는 (특징번호, 임계값, 왼쪽, 오른쪽)
Node = Dict[str, Any]


def _candidate_splits(matrix: Any, bins: int) -> List[Any]:
    """특징별 분할 후보 임계값 (분위수)."""

    numpy = _require_numpy()
    quantiles = numpy.linspace(0.0, 1.0, bins + 2)[1:-1]
    out = []
    for column in range(matrix.shape[1]):
        values = numpy.unique(numpy.quantile(matrix[:, column], quantiles))
        out.append(values)
    return out


def _grow(
    matrix: Any,
    residual: Any,
    rows: Any,
    depth: int,
    candidates: Sequence[Any],
    min_leaf: int,
) -> Node:
    """제곱손실을 가장 많이 줄이는 축 정렬 분할을 재귀적으로 찾는다."""

    numpy = _require_numpy()
    target = residual[rows]
    leaf_value = float(target.mean()) if len(target) else 0.0
    if depth == 0 or len(rows) < 2 * min_leaf:
        return {"value": leaf_value}

    total = target.sum()
    count = len(target)
    best = None
    for column, thresholds in enumerate(candidates):
        if len(thresholds) == 0:
            continue
        values = matrix[rows, column]
        for threshold in thresholds:
            mask = values <= threshold
            left_n = int(mask.sum())
            right_n = count - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            left_sum = float(target[mask].sum())
            # 제곱손실 감소량 = 왼쪽합²/n_L + 오른쪽합²/n_R − 전체합²/n
            gain = (
                left_sum * left_sum / left_n
                + (total - left_sum) ** 2 / right_n
                - total * total / count
            )
            if best is None or gain > best[0]:
                best = (gain, column, float(threshold), mask)

    if best is None or best[0] <= 0.0:
        return {"value": leaf_value}

    _gain, column, threshold, mask = best
    left_rows = rows[mask]
    right_rows = rows[~mask]
    return {
        "feature": int(column),
        "threshold": threshold,
        "left": _grow(matrix, residual, left_rows, depth - 1, candidates, min_leaf),
        "right": _grow(matrix, residual, right_rows, depth - 1, candidates, min_leaf),
    }


def _predict_tree(matrix: Any, node: Node) -> Any:
    numpy = _require_numpy()
    if "value" in node:
        return numpy.full(matrix.shape[0], node["value"])
    mask = matrix[:, node["feature"]] <= node["threshold"]
    out = numpy.empty(matrix.shape[0])
    if mask.any():
        out[mask] = _predict_tree(matrix[mask], node["left"])
    if (~mask).any():
        out[~mask] = _predict_tree(matrix[~mask], node["right"])
    return out


def fit_boosted(
    matrix: Any,
    target: Any,
    *,
    trees: int,
    depth: int,
    learning_rate: float,
    bins: int,
    min_leaf: int,
) -> Dict[str, Any]:
    """제곱손실 부스팅. 반환값이 그대로 직렬화 가능한 모델이다."""

    numpy = _require_numpy()
    base = float(target.mean())
    prediction = numpy.full(matrix.shape[0], base)
    candidates = _candidate_splits(matrix, bins)
    all_rows = numpy.arange(matrix.shape[0])

    forest = []
    for _ in range(trees):
        residual = target - prediction
        tree = _grow(matrix, residual, all_rows, depth, candidates, min_leaf)
        prediction = prediction + learning_rate * _predict_tree(matrix, tree)
        forest.append(tree)
    return {"base": base, "learning_rate": learning_rate, "trees": forest}


def predict(matrix: Any, model: Mapping[str, Any]) -> Any:
    numpy = _require_numpy()
    out = numpy.full(matrix.shape[0], model["base"])
    for tree in model["trees"]:
        out = out + model["learning_rate"] * _predict_tree(matrix, tree)
    return out


# ── 순수 파이썬 추론 — 컨테이너에서 도는 것 ──────────────────────────────────


def predict_row_pure(features: Sequence[float], model: Mapping[str, Any]) -> float:
    """numpy 없이 한 행을 예측한다. 컨테이너 경로가 이 함수를 쓴다."""

    total = model["base"]
    rate = model["learning_rate"]
    for tree in model["trees"]:
        node = tree
        while "value" not in node:
            node = (
                node["left"]
                if features[node["feature"]] <= node["threshold"]
                else node["right"]
            )
        total += rate * node["value"]
    return total


def verify_pure_python(
    matrix: Any, model: Mapping[str, Any], *, tolerance: float = 1e-9
) -> Tuple[bool, float]:
    """numpy 예측과 순수 파이썬 순회를 대조한다. 배포 전 필수."""

    numpy = _require_numpy()
    fast = predict(matrix, model)
    slow = numpy.asarray(
        [predict_row_pure([float(v) for v in row], model) for row in matrix]
    )
    gap = float(numpy.abs(fast - slow).max())
    return gap <= tolerance, gap
