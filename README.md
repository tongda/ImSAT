# Train

`python3 train.py --train-steps 1000 --steps-per-eval 5000 --batch-size 16 --learning-rate 0.00001 --selector --dropout --ctx2out --prev2out --hard-attention --dataset challenger.ai --eval-steps 1000`

# Predict

`python3 predict.py --model-dir ./ckp-dir/selector_True-dropout_True-ctx2out_True-prev2out_True-lr_0.0001/ --batch-size 64 --selector --dropout --ctx2out --prev2out --hard-attention`