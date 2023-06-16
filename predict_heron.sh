#!/bin/bash
python predict_heron.py \
--num_classes=2 \
--model=ewasr_resnet18 \
--dataset-path=/home/tony/Downloads/heron_test \
--weights=/home/tony/eWaSR/checkpoints/epoch=epoch=62-mIoU=val/iou/water=0.99.ckpt \
--output_dir=/home/tony/Downloads/etest \
--batch_size=1 
