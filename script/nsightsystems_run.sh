#!/bin/bash

cd /TRT2022_VilBERT/src

nsys profile -o VilBertModel --stats=true python testVilBertTrt_v1.py > /TRT2022_VilBERT/logs/nsys_layernorm.txt