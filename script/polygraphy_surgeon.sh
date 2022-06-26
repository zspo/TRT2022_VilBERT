#!/bin/bash

cd /TRT2022_VilBERT/models

polygraphy surgeon sanitize \
           vilbert_model_vision_logit.onnx \
           --fold-constant \
           -o \
           vilbert_model_vision_logit_surgeon.onnx \
           > /TRT2022_VilBERT/logs/result-surgeon.txt