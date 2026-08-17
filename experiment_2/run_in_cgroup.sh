#!/bin/bash

CGROUP="/sys/fs/cgroup/$1"
shift

echo $$ > "$CGROUP/cgroup.procs"
exec "$@"