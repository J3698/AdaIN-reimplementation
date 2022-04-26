#!/usr/bin/env bash


python3 main.py "test2" --batch-size 32 --image-size 128 --scheduler-step 5 --scheduler-gamma 0.5 --lambda-style 0.3 --encoder resnet --swag --lr 0.001 --lambda-content 1e5
