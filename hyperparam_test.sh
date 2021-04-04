#!/usr/bin/env bash


# python3 train.py --lambda-style 50 --num-epochs 800 --dataset-length 1 --save-freq 100 --crop "25crop"
# python3 train.py --lambda-style 100 --num-epochs 800 --dataset-length 1 --save-freq 100 --crop "50crop"
# python3 train.py --lambda-style 200 --num-epochs 800 --dataset-length 1 --save-freq 100 --crop "100crop"

python3 train.py --crop --lambda-style 50 --num-epochs 100000 --dataset-length 10000 --save-freq 10 ".50.10k.bigboi"

python3 train.py --lambda-style 25 --num-epochs 800 --dataset-length 1 --save-freq 100 "25ncrop"
python3 train.py --lambda-style 50 --num-epochs 800 --dataset-length 1 --save-freq 100 "50ncrop"
python3 train.py --lambda-style 100 --num-epochs 800 --dataset-length 1 --save-freq 100 "100ncrop"


