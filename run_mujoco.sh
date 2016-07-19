#!/bin/sh

echo_and_run() { echo "$@"; $@; }

for env in "InvertedPendulum-v1" "InvertedDoublePendulum-v1" "Reacher-v1," "HalfCheetah-v1" "Swimmer-v1" "Hopper-v1" "Walker2d-v1" "Ant-v1" "Ant-v1" "HumanoidStandup-v1"; do
  echo_and_run xvfb-run --auto-servernum -s "-screen 0 1400x900x24" python main.py --env_name=$env &
done
