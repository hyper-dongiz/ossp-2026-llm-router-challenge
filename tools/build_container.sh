#!/usr/bin/env bash
# 이 저장소만 clone 하면 언제든 제출 이미지를 만들고 검증한다. 자격증명이 필요 없다.
#
#   ./tools/build_container.sh            빌드 + 검증
#   ./tools/build_container.sh --no-verify  빌드만
#
# 공식 실행 규격은 docs/RUNTIME.md, 제출 요건은 docs/SUBMISSION.md 를 따른다.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-ossp-router:submit}
VERIFY=1
[ "${1:-}" = "--no-verify" ] && VERIFY=0

SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=""
[ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY=" (작업 트리 변경 있음)"

echo "▶ 빌드 · linux/arm64 · 커밋 ${SHA}${DIRTY}"
docker build --pull --platform linux/arm64 \
  --file container/Dockerfile \
  --tag "$TAG" --tag "ossp-router:$SHA" .

echo "▶ 이미지"
docker image inspect "$TAG" --format '  ID {{.Id}}'
docker image inspect "$TAG" --format '  크기 {{.Size}} bytes'

[ "$VERIFY" = "1" ] || { echo "검증 생략"; exit 0; }

# 공개 데이터가 없으면 검증을 건너뛴다
IN=data/materialized/dev/inputs.json
if [ ! -f "$IN" ]; then
  echo "▶ 검증 생략 — $IN 이 없다. 아래를 먼저 실행하라"
  echo "    python3 -m venv .venv-data"
  echo "    .venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt"
  echo "    .venv-data/bin/python tools/materialize_public_data.py"
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/in" "$WORK/out"
cp "$IN" "$WORK/in/inputs.json"
# mktemp 는 700 으로 만든다. 비루트(UID 65532) 컨테이너가 바인드 마운트를
# 읽고 출력을 쓸 수 있도록 권한을 연다.
chmod 755 "$WORK" "$WORK/in"
chmod 644 "$WORK/in/inputs.json"
chmod 777 "$WORK/out"

echo "▶ 공식 자원 한도로 세 등급 실행"
for t in fast balanced premium; do
  S=$(python3 -c 'import time;print(int(time.time()*1000))')
  docker run --rm --platform linux/arm64 \
    --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 \
    --network none --read-only --tmpfs /tmp:size=256m \
    -v "$WORK/in:/challenge/input:ro" -v "$WORK/out:/challenge/output" \
    "$TAG" --input /challenge/input/inputs.json \
    --tier "$t" --output /challenge/output/submission.json
  E=$(python3 -c 'import time;print(int(time.time()*1000))')
  mv "$WORK/out/submission.json" "$WORK/out/$t.json"
  echo "  $t  $(( (E-S) )) ms  한도 90000 ms"
done

echo "▶ 출력 스키마"
PYTHONPATH=src python3 - "$WORK/out" <<'PY'
import json, sys, pathlib
from ossp_router.protocol import parse_submission
out = pathlib.Path(sys.argv[1])
for t in ('fast', 'balanced', 'premium'):
    d = json.loads((out / f'{t}.json').read_text(encoding='utf-8'))
    parse_submission(d)
    print(f'  {t}  decisions {len(d["decisions"])}  스키마 통과')
PY

echo "▶ 공식 채점기"
PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input "$IN" --outcomes data/dev/outcomes.json \
  --submissions "$WORK/out" --report "$WORK/report.json" >/dev/null
python3 - "$WORK/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('  final_score', r['final_score'])
for t in ('fast', 'balanced', 'premium'):
    d = r['tiers'][t]
    print(f'  {t}  ratio {d["budget_ratio"][:14]}  passed {d["budget_passed"]}')
PY

echo "✔ 빌드와 검증 완료"
