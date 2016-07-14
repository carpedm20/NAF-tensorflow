#!/bin/sh

echo_and_run() { echo "$@"; $@; }

for env in "InvertedPendulum-v1" "InvertedDoublePendulum-v1" "REACHER-V1" "HALFCHEETAH-V1" "SWIMMER-V1" "HOPPER-V1" "WALKER2D-V1" "ANT-V1" "HUMANOID-V1" "HUMANOIDSTANDUP-V1"; do
  echo_and_run python main.py --env_name=$env &
done
