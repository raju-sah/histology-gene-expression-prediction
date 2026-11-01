#!/usr/bin/env bash
# Run a command with an RSS cap (the local machine is shared and has little free RAM) and a disk guard.
# Usage: scripts/run_with_watchdog.sh LOGFILE LIMIT_KB command [args...]
# Output goes to LOGFILE; the process is killed if its resident memory exceeds LIMIT_KB, or (disk guard) if free
# disk on the working directory's filesystem drops below MIN_FREE_GIB (default 5). Free disk and the size of
# data/tmp_fitcache are logged to LOGFILE every DISK_LOG_SEC seconds (default 180) to tell our run from other processes.
LOG=$1; LIMIT=$2; shift 2
MIN_FREE_GIB=${MIN_FREE_GIB:-5}
DISK_LOG_SEC=${DISK_LOG_SEC:-180}
MIN_FREE_BYTES=$(( MIN_FREE_GIB * 1024 * 1024 * 1024 ))
"$@" > "$LOG" 2>&1 &
P=$!
PEAK=0
LAST_DISK=0
while kill -0 $P 2>/dev/null; do
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); R=${R:-0}
  [ "$R" -gt "$PEAK" ] && PEAK=$R
  if [ "$R" -gt "$LIMIT" ]; then echo "WATCHDOG: killed at ${R} kB" >> "$LOG"; kill $P; fi
  NOW=$(date +%s)
  if [ $(( NOW - LAST_DISK )) -ge "$DISK_LOG_SEC" ]; then
    LAST_DISK=$NOW
    FREE=$(df -B1 --output=avail . | tail -1 | tr -d ' ')
    CACHE=$(du -sm data/tmp_fitcache 2>/dev/null | cut -f1)
    echo "DISKLOG $(date +%H:%M:%S) free_bytes=$FREE free_gib=$(( FREE / 1073741824 )) fitcache_mb=${CACHE:-0} rss_kb=$R" >> "$LOG"
    if [ "$FREE" -lt "$MIN_FREE_BYTES" ]; then
      echo "WATCHDOG: DISK GUARD stopping run, free ${FREE} B < ${MIN_FREE_GIB} GiB" >> "$LOG"
      kill -TERM $P
    fi
  fi
  sleep 5
done
wait $P
RC=$?
echo "EXIT $RC PEAK_RSS_KB $PEAK" >> "$LOG"
exit $RC
