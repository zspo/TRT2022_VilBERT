#!/bin/bash

cd /TRT2022_VilBERT/models

nsys profile -o test_model trtexec --loadEngine=./vilbert_model_vision_logit.plan --warmUp=0 --duration=0 --iterations=50

nsys profile -o test_model_ln trtexec --loadEngine=./vilbert_model_vision_logit_layernorm.plan --warmUp=0 --duration=0 --iterations=50 --plugins=../libs/LayerNormPlugin.so
