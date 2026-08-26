#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 이승재
# SPDX-License-Identifier: Apache-2.0
#
# 이미지를 공개 레지스트리에 push 하고 submission-ossp-skt.json 을 만든다.
# 사용법: REGISTRY=ghcr.io/hyper-dongiz USER=hyper-dongiz TOKEN_FILE=~/.reg-token ./tools/push_and_submit.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${REGISTRY:?REGISTRY 를 지정하라. 예 ghcr.io/hyper-dongiz 또는 docker.io/사용자명}"
: "${USER:?USER 를 지정하라}"
: "${TOKEN_FILE:?TOKEN_FILE 을 지정하라}"

SHA=$(git rev-parse HEAD)
[ -z "$(git status --porcelain)" ] || { echo "작업 트리가 깨끗하지 않다. 커밋부터 하라"; exit 1; }

HOST=${REGISTRY%%/*}
REPO="$REGISTRY/ossp-router"

echo "▶ 로그인 $HOST"
tr -d '\n' < "$TOKEN_FILE" | docker login "$HOST" -u "$USER" --password-stdin

echo "▶ 빌드 linux/arm64 · 커밋 $SHA"
docker build --platform linux/arm64 --file container/Dockerfile \
  --tag "$REPO:$SHA" --tag "$REPO:submit" .

echo "▶ push"
docker push "$REPO:$SHA"

DIGEST=$(docker buildx imagetools inspect "$REPO:$SHA" --format '{{.Manifest.Digest}}' 2>/dev/null \
  || docker image inspect "$REPO:$SHA" --format '{{index .RepoDigests 0}}' | sed 's/.*@//')
[ -n "$DIGEST" ] || { echo "다이제스트를 못 얻었다"; exit 1; }

echo "▶ submission-ossp-skt.json"
python3 - "$SHA" "$REPO" "$DIGEST" <<'PY'
import json, sys
sha, repo, digest = sys.argv[1:4]
doc = {
    "schema_version": 1,
    "challenge_id": "ossp-2026-llm-router-challenge",
    "repository_url": "https://github.com/hyper-dongiz/ossp-2026-llm-router-challenge",
    "commit_sha": sha,
    "image_digest": f"{repo}@{digest}",
    "primary_license": "Apache-2.0",
}
with open("submission-ossp-skt.json", "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(json.dumps(doc, ensure_ascii=False, indent=1))
PY

echo "▶ 스키마 검증"
python3 tools/validate_technical_submission.py

echo "▶ 남은 것 — 아래를 실행하라"
echo "  git add submission-ossp-skt.json"
echo "  git commit -m 'submission-ossp-skt.json 추가'"
echo "  git push origin main"
echo "  결과보고서의 프로젝트 등록 URL = https://github.com/hyper-dongiz/ossp-2026-llm-router-challenge/tree/<그 커밋 SHA>"
