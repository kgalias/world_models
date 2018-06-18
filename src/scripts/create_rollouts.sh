#!/usr/bin/env bash

xvfb-run -a -s "-screen 0 1400x900x24" python src/rollout.py --env CarRacing-v0 --agent RandomAgent --n_rollouts 10000
