#!/bin/zsh
# Wait for the running jewelry batch to finish, then run clippers, then lotions.
cd /Users/chrisilias/HICLONE/picture-upload

echo "[queue] waiting for jewelry run to finish..."
while pgrep -f "app.batch --root incoming/jewelry" >/dev/null; do sleep 20; done
echo "[queue] jewelry finished at $(date)"

for cat in clippers lotions; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="logs/${cat}_${ts}.log"
  echo "[queue] starting $cat -> $log"
  python3 -m app.batch --root "incoming/$cat" --mode variants --all --creative both > "$log" 2>&1
  echo "[queue] $cat done at $(date) (exit $?)"
done
echo "[queue] all done at $(date)"
