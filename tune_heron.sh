#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python train.py \
    --heron \
    --validation \
    --num_classes=2 \
    --gpus=1 \
    --model=ewasr_resnet18 \
    --model_name=heron \
    --log_steps=2 \
    --batch_size=3 \
    --epochs=150 \
    --pretrained_weights=/home/tony/Downloads/ewasr_resnet18.pth \
    --heron_data_dir=/home/tony/train
