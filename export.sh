#!/bin/bash

python export.py \
--architecture=ewasr_resnet18 \
--num_classes=2 \
--weights-file=/home/tony/eWaSR/output/logs/heron/version_9/checkpoints/epoch90.ckpt \
--output-dir=/home/tony/Downloads \
--onnx_only


