#!/usr/bin/env bash

xvfb-run -a -s "-screen 0 1400x900x24" python src/train_controller.py --vae_fname 2018-06-26T08:09:18.665709100 --eval_interval 5
