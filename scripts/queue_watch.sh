#!/usr/bin/env bash
# Operating rule (CLAUDE.md §12): exit (= notify) when (a) a live job's log has not changed for LIMIT s, or
# (b) any GPU has no compute process (queue empty) for IDLE s. usage: queue_watch.sh <limit_s> <idle_s> <pidfile:logfile> ...
LIMIT=$1; IDLE=$2; shift 2
declare -A idle_since
while true; do
  for pl in "$@"; do
    pf=${pl%%:*}; lf=${pl#*:}; p=$(cat $pf 2>/dev/null); kill -0 $p 2>/dev/null || continue
    [ -f "$lf" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$lf") ))
    [ $age -gt $LIMIT ] && { echo "STALL: $lf unchanged ${age}s (pid $p) $(date +%H:%M)"; exit 0; }
  done
  for g in ${GPUS:-0 1 2 3}; do  # GPUS="0 1" watches only the GPUs this queue uses
    n=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -vc "^$")
    viewer=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | while read q; do ps -o args= -p $q | grep -c viewer4d; done | paste -sd+ | bc 2>/dev/null)
    busy=$(( n - ${viewer:-0} ))
    if [ $busy -le 0 ]; then
      [ -z "${idle_since[$g]}" ] && idle_since[$g]=$(date +%s)
      [ $(( $(date +%s) - ${idle_since[$g]} )) -gt $IDLE ] && { echo "GPU$g IDLE (queue empty) since $(date -d @${idle_since[$g]} +%H:%M)"; exit 0; }
    else unset idle_since[$g]; fi
  done
  sleep 60
done
