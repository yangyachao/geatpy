#!/bin/bash
# Regenerate tests/fixtures/core_parity_orig_2_7_0.json from the official geatpy 2.7.0 binaries.
# Usage: run_original_core.sh [--reps N] [--skip-templates --merge]
# Requires docker (linux/amd64 emulation is fine) and the upstream history (commit 49a1d23f).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ORIG_COMMIT=49a1d23f

docker build -q --platform linux/amd64 -t geatpy-orig:2.7 "$ROOT/scripts/parity" >/dev/null
# Working-tree Python layer + original 2.7.0 core binaries.
mkdir -p "$WORK/tests" "$WORK/scripts/parity"
cp -R "$ROOT/geatpy" "$WORK/"
cp "$ROOT/tests/parity_cases.py" "$WORK/tests/"
cp "$ROOT/scripts/parity/gen_golden.py" "$WORK/scripts/parity/"
git -C "$ROOT" archive "$ORIG_COMMIT" _core/Linux/lib64/v3.6 | tar -x -C "$WORK"
rm -rf "$WORK/geatpy/core" && mkdir "$WORK/geatpy/core"
cp "$WORK"/_core/Linux/lib64/v3.6/*.so "$WORK/geatpy/core/" && touch "$WORK/geatpy/core/__init__.py"
FIXTURE="$ROOT/tests/fixtures/core_parity_orig_2_7_0.json"
[ -f "$FIXTURE" ] && cp "$FIXTURE" "$WORK/out.json"
docker run --rm --platform linux/amd64 -e PYTHONPATH=/w -v "$WORK":/w -w /w geatpy-orig:2.7 \
    python scripts/parity/gen_golden.py /w/out.json "$@"
cp "$WORK/out.json" "$FIXTURE"
echo "wrote tests/fixtures/core_parity_orig_2_7_0.json"
