#!/bin/bash

DISPLAY=1
MODEL_PATH="./model"
INIT_MODEL="./model/final.pth"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

if [ "$DISPLAY" -eq "1" ]; then
    CUDA_VISIBLE_DEVICES=1 python main.py --test --init_model=$INIT_MODEL --motion_range=2 --display 2>&1 | tee $MODEL_PATH/test.log
else
    CUDA_VISIBLE_DEVICES=1 python main.py --test --init_model=$INIT_MODEL --motion_range=2 2>&1 | tee $MODEL_PATH/test.log
fi

cp test.sh $MODEL_PATH
