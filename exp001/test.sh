#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py \
  --test \
  --init_model=./model/final.pth \
  --motion_range=3 2>&1 | tee $MODEL_PATH/test.log

cp test.sh $MODEL_PATH/test.sh
