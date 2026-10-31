#!/usr/bin/env bash
# Run a command with an RSS cap (the local machine is shared and has little free RAM).
# Usage: scripts/run_with_watchdog.sh LOGFILE LIMIT_KB command [args...]
# Output goes to LOGFILE; the process is killed if its resident memory exceeds LIMIT_KB.
LOG=$1; LIMIT=$2; shift 2
"$@" > "$LOG" 2>&1 &
P=$!
PEAK=0
while kill -0 $P 2>/dev/null; do
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); R=${R:-0}
  [ "$R" -gt "$PEAK" ] && PEAK=$R
  if [ "$R" -gt "$LIMIT" ]; then echo "WATCHDOG: killed at ${R} kB" >> "$LOG"; kill $P; fi
  sleep 5
done
wait $P
RC=$?
echo "EXIT $RC PEAK_RSS_KB $PEAK" >> "$LOG"
exit $RC
