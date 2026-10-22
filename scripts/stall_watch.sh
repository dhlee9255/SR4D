#!/usr/bin/env bash
# Exit (= notify) as soon as any watched log has not changed for LIMIT seconds while its PID is alive.
# usage: stall_watch.sh <limit_s> <pidfile:logfile> ...
LIMIT=$1; shift
while true; do
  alive=0
  for pl in "$@"; do
    pf=${pl%%:*}; lf=${pl#*:}; p=$(cat $pf 2>/dev/null); kill -0 $p 2>/dev/null || continue; alive=1
    [ -f "$lf" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$lf") ))
    if [ $age -gt $LIMIT ]; then echo "STALL: $lf unchanged for ${age}s (pid $p) at $(date +%H:%M)"; exit 0; fi
  done
  [ $alive = 0 ] && { echo "all watched runs finished $(date +%H:%M)"; exit 0; }
  sleep 120
done
