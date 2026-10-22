#!/usr/bin/env bash
# dance D-025 pose runs after the current dance jobs on GPU1 (sr4d_v0) and GPU3 (4dgs -> ftgspp). Waits by PID file.
cd "$(dirname "$0")/.."
G=$1; V=$2; PIDF=$3
while kill -0 $(cat $PIDF) 2>/dev/null; do sleep 60; done
scripts/run_r15h35.sh dance $V $G
