# SPDX-FileCopyrightText: Copyright 2026 hyper-dongiz
# SPDX-License-Identifier: Apache-2.0

"""제출 라우터 — 컨테이너 진입점이 호출하는 것.

운영자는 `router-run --input ... --tier ... --output ...` 으로만 부른다. 아티팩트
경로는 인자로 오지 않으므로 이미지에 동봉된 것을 쓴다
(`ossp_router/resources/router-artifact.v1.json`).

## 추론은 왜 새로 쓰지 않는가

우리 정책은 공개 hash-regex 와 **같은 아티팩트 스키마**에 담긴다(`analysis/bake_artifact.py`).
계수와 안전계수만 우리 것이고 계산 절차는 동일하므로, 추론 코드를 복제하지 않고
`hash_regex` 모듈을 그대로 재사용한다. 표준 라이브러리만 쓰므로 컨테이너 제약을 만족한다.

정책의 근거와 측정값은 `docs/EVALUATION.md` 를 보라.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from ossp_router.heuristic import write_submission_atomic
from ossp_router.protocol import (
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_policy,
)

ARTIFACT_PATH = Path(__file__).resolve().parent / "resources" / "router-artifact.v1.json"


def _hash_regex() -> Any:
    """추론 모듈을 찾는다.

    이미지에서는 `hash_regex` 가 최상위 모듈로 놓이고(Dockerfile 이 그렇게 복사한다),
    개발 저장소에서는 `baselines.hash_regex` 다. 둘 다 같은 파일이다.
    """

    try:
        import hash_regex  # type: ignore[import-not-found]

        return hash_regex
    except ModuleNotFoundError:
        from baselines import hash_regex as module

        return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="hyper-dongiz 제출 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ARTIFACT_PATH,
        help="기본값은 이미지에 동봉된 아티팩트. 실험용으로만 바꾼다",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    hash_regex = _hash_regex()
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        artifact = hash_regex.load_artifact(args.artifact)
        plan = hash_regex.make_hash_regex_submission(
            inputs, policy, artifact, args.tier
        )
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: {args.tier} 제출 파일을 생성했습니다 "
        f"(예측 비용 비율 {plan.predicted_budget_ratio:.6f}, "
        f"안전계수 {plan.safety_ratio:.4f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
