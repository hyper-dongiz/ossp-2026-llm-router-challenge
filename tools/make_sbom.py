#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 이승재
# SPDX-License-Identifier: Apache-2.0

"""제출 이미지의 CycloneDX SBOM 을 만든다.

이미지 안에서 `apk info` 와 `apk info -L` 를 직접 읽어 구성 요소를 뽑는다.
외부 스캐너를 쓰지 않는 이유는 결과를 이 저장소만으로 재현할 수 있게 하려는
것이다. 이미지에는 파이썬 외부 패키지가 없다. `pip` · `setuptools` · `wheel`
은 빌드 단계에서 제거하고 실행 코드는 표준 라이브러리만 쓴다.

실행
    python3 tools/make_sbom.py --tag ossp-router:submit --out container/sbom.cdx.json
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMAGE = 'python:3.11.15-alpine3.23'
BASE_DIGEST = ('sha256:f73754c398b259dfbbe482361dca8b464dea57da74efe5214966'
               'ca2ee767ee12')


def sh(tag, command):
    out = subprocess.run(
        ['docker', 'run', '--rm', '--platform', 'linux/arm64', '--network', 'none',
         '--entrypoint', '/bin/sh', tag, '-c', command],
        capture_output=True, text=True, check=True)
    return out.stdout


def image_digest(tag):
    out = subprocess.run(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


_APK_RELEASE = re.compile(r'^r\d+$')


def apk_components(tag):
    """`apk info -v` 한 줄이 `이름-버전-릴리스` 형태다.

    릴리스 칸이 `r<숫자>` 일 때만 두 번 잘라 이름과 버전을 나눈다.
    `.python-rundeps-20260706.220128` 같은 가상 패키지는 릴리스 칸이 없어
    한 번만 자른다.
    """
    lines = [l.strip() for l in sh(tag, 'apk info -v').splitlines() if l.strip()]
    out = []
    for line in lines:
        head, _, last = line.rpartition('-')
        if head and _APK_RELEASE.match(last):
            pkg, _, ver = head.rpartition('-')
            version = f'{ver}-{last}'
            if not pkg:
                pkg, version = head, last
        elif head:
            pkg, version = head, last
        else:
            pkg, version = line, ''
        out.append({
            'type': 'library',
            'name': pkg,
            'version': version,
            'purl': f'pkg:apk/alpine/{pkg}@{version}?arch=aarch64' if version
                    else f'pkg:apk/alpine/{pkg}',
            'scope': 'required',
        })
    return out


def source_components():
    """이미지에 복사한 우리 소스 파일. 경로와 SHA-256 을 남긴다."""
    files = []
    for rel in ('src/ossp_router/submission_router.py',
                'src/ossp_router/resources/router-artifact.v1.json',
                'baselines/hash_regex.py',
                'container/entrypoint.py'):
        path = os.path.join(ROOT, rel)
        with open(path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        files.append({
            'type': 'file',
            'name': rel,
            'hashes': [{'alg': 'SHA-256', 'content': digest}],
            'licenses': [{'license': {'id': 'Apache-2.0'}}],
        })
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ossp-router:submit')
    ap.add_argument('--out', default=os.path.join(ROOT, 'container/sbom.cdx.json'))
    a = ap.parse_args()

    components = [{
        'type': 'container',
        'name': BASE_IMAGE,
        'version': '3.11.15-alpine3.23',
        'purl': f'pkg:docker/library/python@{BASE_DIGEST}',
        'hashes': [{'alg': 'SHA-256', 'content': BASE_DIGEST.split(':', 1)[1]}],
        'description': '기반 이미지. container/BASE_IMAGE.md 에 출처와 조리법을 기록했다',
    }]
    components.extend(apk_components(a.tag))
    components.extend(source_components())

    doc = {
        'bomFormat': 'CycloneDX',
        'specVersion': '1.5',
        'version': 1,
        'metadata': {
            'component': {
                'type': 'container',
                'name': 'ossp-router',
                'version': subprocess.run(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                    capture_output=True, text=True).stdout.strip(),
                'licenses': [{'license': {'id': 'Apache-2.0'}}],
            },
            'properties': [
                {'name': 'image.id', 'value': image_digest(a.tag)},
                {'name': 'image.platform', 'value': 'linux/arm64'},
                {'name': 'python.external_packages', 'value': '없음'},
                {'name': 'ai.model', 'value':
                 '해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음'},
            ],
        },
        'components': components,
    }
    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, sort_keys=False)
        f.write('\n')
    os.replace(tmp, a.out)
    print(f'구성 요소 {len(components)}개 -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
