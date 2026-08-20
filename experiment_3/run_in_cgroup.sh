#!/bin/bash

CGROUP="/sys/fs/cgroup/$1"
shift

echo $$ > "$CGROUP/cgroup.procs"
exec "$@"

sudo ./run_in_cgroup.sh three_p ./p1 & sudo ./run_in_cgroup.sh three_p ./p2 & sudo ./run_in_cgroup.sh three_p ./p3

python3 generate_mem_graph.py \
  --case "16GB Hard Limit:run16.csv:127967,127966,127968:16" \
  --case "14GB Hard Limit:run14.csv:128803,128805,128804:14" \
  --case "10GB Hard Limit:run10.csv:129112,129113,129114:10" \
  --case "7GB Hard Limit:run7.csv:129361,129359,129360:7" \
  --out resident_memory_x_hard_limits.png