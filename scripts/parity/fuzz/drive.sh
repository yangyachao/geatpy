#!/bin/bash
# usage: [SCRIPT=fz2.py] drive.sh PYTHON OUT.jsonl [REPS_SCALE]
# Runs fz.py (or $SCRIPT) with PYTHON until every case is recorded in OUT.jsonl. A case that
# crashes the interpreter or runs over 300 s is recorded as {"status": "crash"} and skipped.
PY=$1; OUT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); SC=${3:-1}
cd "$(dirname "$0")"
for i in $(seq 1 200); do
  PYTHONPATH= perl -e "alarm shift; exec @ARGV" 300 "$PY" -I "${SCRIPT:-fz.py}" "$OUT" "$SC" 2>>"$OUT.stderr"; rc=$?
  [ $rc -eq 0 ] && exit 0
  last=$(tail -1 "$OUT")
  if [[ $last == START* ]]; then
    n=${last#START }
    echo "{\"name\": \"$n\", \"status\": \"crash\", \"error\": \"exit code $rc\"}" >> "$OUT"
  fi
done
