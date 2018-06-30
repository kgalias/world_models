#!/usr/bin/env bash

xvfb-run -a -s "-screen 0 1400x900x24" python src/rollout.py
