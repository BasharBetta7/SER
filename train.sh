#!/bin/sh
python ./src/run.py --config ./configs/config.yaml --batch_size 2 --epochs 20 --learning_rate 4e-5 --accum_grad 4 --cross_val --early_stop --num_folds 5