#!/usr/bin/env bash
set -e

# Run collect_data.py for 50 episodes in a row
for i in $(seq 1 20); do
  echo "=== Starting episode $i ($(date +'%Y-%m-%d %H:%M:%S')) ==="
  python collect_data.py
  echo "=== Finished  episode $i ($(date +'%Y-%m-%d %H:%M:%S')) ==="
done
