#!/bin/bash

# Create directories if not exist
MODEL_PATH=./model
if [[ ! -e $MODEL_PATH ]]; then
    mkdir -p $MODEL_PATH
else
    echo "$MODEL_PATH already exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py --train --method=supervised --train_epoch=100 --test_interval=50 2>&1 | tee $MODEL_PATH/train.log

cp train_s.sh $MODEL_PATH/train.sh
