#!/usr/bin/env bash
# Background launcher for generate_answers.py - start it, log out, come back later.
#
# Usage:
#   bash run.sh start        # launch the reranked-retrieval run in the background
#   bash run.sh status       # is it running? how many of the 50 rows are done?
#   bash run.sh tail         # follow the live log (Ctrl+C stops following, not the run)
#   bash run.sh stop         # kill the run (completed rows are already saved)
#
# The run survives SSH logout (nohup + its own process group via setsid). Progress is
# visible two ways: the log gets one tqdm line per completed query, and
# results/answers_hybrid_rerank.jsonl grows one row per completed query (flushed
# immediately). Only one config here (not four), so no --model-size or config argument
# is needed - unlike ../generation_cluster/run.sh.

set -u
cd "$(dirname "$0")"

PID_FILE="run.pid"
LOG_DIR="logs"
TOTAL_ROWS=50
RESULTS="results/answers_hybrid_rerank.jsonl"

pid_field()  { cut -d' ' -f"$1" "$PID_FILE"; }
is_running() { [ -f "$PID_FILE" ] && kill -0 "$(pid_field 1)" 2>/dev/null; }

case "${1:-}" in
  start)
    if is_running; then
      echo "Already running (PID $(pid_field 1)). Use: bash run.sh status"
      exit 1
    fi
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/rerank_$(date +%Y-%m-%d_%H-%M-%S).log"
    setsid nohup uv run python generate_answers.py >"$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID $LOG_FILE" > "$PID_FILE"
    echo "Started in background: PID $PID"
    echo "Log: $LOG_FILE"
    echo "Safe to log out. Check back with: bash run.sh status"
    ;;

  status)
    if [ ! -f "$PID_FILE" ]; then
      echo "No run recorded (no $PID_FILE). Start one with: bash run.sh start"
      exit 1
    fi
    LOG_FILE=$(pid_field 2)
    if is_running; then
      echo "RUNNING (PID $(pid_field 1))"
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
    tail -f "$(pid_field 2)"
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
    echo "Usage: bash run.sh {start | status | tail | stop}"
    exit 1
    ;;
esac
