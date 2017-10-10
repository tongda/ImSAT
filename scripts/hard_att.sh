#!/usr/bin/env bash

python3 train.py --train-steps 30000 --steps-per-eval 5000 --batch-size 8 --learning-rate 0.0001 --selector --dropout --ctx2out --prev2out --dataset challenger.ai --eval-steps 200 --hard-attention

python3 train.py --train-steps 100000 --steps-per-eval 1000 --batch-size 8 --learning-rate 0.0001 --selector --dropout --ctx2out --prev2out --dataset challenger.ai --eval-steps 200 --hard-attention --use-sampler