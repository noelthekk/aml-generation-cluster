#!/usr/bin/env bash
# Background launcher for generate_answers.py - start it, log out, come back later.
#
# Usage:
#   bash run.sh start         # launch the primary 8B run in the background (default)
#   bash run.sh start 70b     # (or the 70B generator-scale ablation)
#   bash run.sh status        # is it running? how many of the 200 rows are done?
#   bash run.sh tail          # follow the live log (Ctrl+C stops following, not the run)
#   bash run.sh stop          # kill the run (completed rows are already saved)
#
# The run survives SSH logout (nohup + its own process group via setsid). Progress is
# visible two ways: the log gets one tqdm line per completed query, and
# results/answers_<size>.jsonl grows one row per completed query (flushed immediately).
#
# Model size is passed straight through to generate_answers.py --model-size, which
# validates it (currently 8b/70b) - not re-validated here, so a new size added there
# doesn't also need updating here.

set -u
cd "$(dirname "$0")"

PID_FILE="run.pid"
LOG_DIR="logs"
TOTAL_ROWS=200

pid_field()  { cut -d' ' -f"$1" "$PID_FILE"; }
is_running() { [ -f "$PID_FILE" ] && kill -0 "$(pid_field 1)" 2>/dev/null; }

case "${1:-}" in
  start)
    if is_running; then
      echo "Already running (PID $(pid_field 1)). Use: bash run.sh status"
      exit 1
    fi
    SIZE="${2:-8b}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/answers_${SIZE}_$(date +%Y-%m-%d_%H-%M-%S).log"
    setsid nohup uv run python generate_answers.py --model-size "$SIZE" >"$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID $SIZE $LOG_FILE" > "$PID_FILE"
    echo "Started in background: PID $PID, model size $SIZE"
    echo "Log: $LOG_FILE"
    echo "Safe to log out. Check back with: bash run.sh status"
    ;;

  status)
    if [ ! -f "$PID_FILE" ]; then
      echo "No run recorded (no $PID_FILE). Start one with: bash run.sh start"
      exit 1
    fi
    SIZE=$(pid_field 2)
    LOG_FILE=$(pid_field 3)
    RESULTS="results/answers_${SIZE}.jsonl"
    if is_running; then
      echo "RUNNING (PID $(pid_field 1), model size $SIZE)"
    else
      echo "NOT RUNNING (finished or died - check the end of the log)"
    fi
    if [ -f "$RESULTS" ]; then
      DONE=$(wc -l < "$RESULTS")
      echo "Progress: $DONE/$TOTAL_ROWS rows in $RESULTS"
    else
      echo "Progress: 0/$TOTAL_ROWS (still loading the model - no results file yet)"
    fi
    echo "Last log line:"
    tail -n 1 "$LOG_FILE"
    ;;

  tail)
    if [ ! -f "$PID_FILE" ]; then
      echo "No run recorded (no $PID_FILE)."
      exit 1
    fi
    tail -f "$(pid_field 3)"
    ;;

  stop)
    if ! is_running; then
      echo "Nothing running."
      exit 1
    fi
    PID=$(pid_field 1)
    kill -- -"$PID" 2>/dev/null || kill "$PID"
    echo "Stopped PID $PID. Completed rows are kept in results/ (flushed per row)."
    ;;

  *)
    echo "Usage: bash run.sh {start [model-size] | status | tail | stop}"
    exit 1
    ;;
esac
