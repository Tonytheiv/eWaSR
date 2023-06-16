#!/bin/bash

MODEL=eWaSR-v1

python export.py \
--architecture=ewasr_resnet18 \
--num_classes=2 \
--weights-file=./ckpt/$MODEL.ckpt \
--output-dir=./ckpt \
--onnx_only

mv ./ckpt/ewasr_resnet18.onnx ./ckpt/$MODEL.ckpt.onnx

/usr/src/tensorrt/bin/trtexec --onnx=./ckpt/$MODEL.ckpt.onnx \
    --minShapes=modelInput:1x3x360x640 \
    --maxShapes=modelInput:1x3x360x640 \
    --buildOnly \
    --best \
    --saveEngine=./ckpt/$MODEL.trt

/usr/src/tensorrt/bin/trtexec --loadEngine=./ckpt/$MODEL.trt
