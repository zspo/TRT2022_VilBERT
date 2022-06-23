#!/bin/bash

cd /TRT2022_VilBERT/vilbert_pytorch
python convert_onnx.py --bert_model bert-base-uncased --from_pretrained /TRT2022_VilBERT/models/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 4