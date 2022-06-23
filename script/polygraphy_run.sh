#!/bin/bash

cd /TRT2022_VilBERT/models

# 使用 .onnx 构建一个 TensorRT 引擎，使用 FP16精度，同时在 onnxruntime 和 TensorRT 中运行，对比结果
# polygraphy run vilbert_model_vision_logit.onnx \
#     --onnxrt --trt \
#     --workspace 20G \
#     --save-engine=vilbert_model_vision_logit_fp16_poly.plan \
#     --atol 1e-3 --rtol 1e-3 \
#     --fp16 \
#     --verbose \
#     --trt-outputs mark all --onnx-outputs mark all \
#     --trt-min-shapes 'question:[1, 20]' 'features:[1, 100, 2048]' 'spatials:[1, 100, 5]' 'segment_ids:[1, 20]' 'input_mask:[1, 20]' 'image_mask:[1, 100]' \
#     --trt-opt-shapes 'question:[16, 20]' 'features:[16, 100, 2048]' 'spatials:[16, 100, 5]' 'segment_ids:[16, 20]' 'input_mask:[16, 20]' 'image_mask:[16, 100]' \
#     --trt-max-shapes 'question:[256, 20]' 'features:[256, 100, 2048]' 'spatials:[256, 100, 5]' 'segment_ids:[256, 20]' 'input_mask:[256, 20]' 'image_mask:[256, 100]' \
#     --input-shapes   'question:[16, 20]' 'features:[16, 100, 2048]' 'spatials:[16, 100, 5]' 'segment_ids:[16, 20]' 'input_mask:[16, 20]' 'image_mask:[16, 100]' \
#     > result-run-FP16_MarkAll.txt

# 注意参数名和格式跟 trtexec 不一样，多个形状之间用空格分隔，如：
# --trt-max-shapes 'input0:[16,320,256]' 'input1:[16，320]' 'input2:[16]'


for outnodes in `seq 540 3800`;
do
echo ${outnodes}

polygraphy run vilbert_model_vision_logit.onnx \
    --onnxrt --trt \
    --workspace 20G \
    --save-engine=vilbert_model_vision_logit_fp16_poly.plan \
    --atol 1e-3 --rtol 1e-3 \
    --fp16 \
    --verbose \
    --trt-outputs ${outnodes} \
    --onnx-outputs ${outnodes} \
    --trt-min-shapes 'question:[1, 20]' 'features:[1, 100, 2048]' 'spatials:[1, 100, 5]' 'segment_ids:[1, 20]' 'input_mask:[1, 20]' 'image_mask:[1, 100]' \
    --trt-opt-shapes 'question:[16, 20]' 'features:[16, 100, 2048]' 'spatials:[16, 100, 5]' 'segment_ids:[16, 20]' 'input_mask:[16, 20]' 'image_mask:[16, 100]' \
    --trt-max-shapes 'question:[256, 20]' 'features:[256, 100, 2048]' 'spatials:[256, 100, 5]' 'segment_ids:[256, 20]' 'input_mask:[256, 20]' 'image_mask:[256, 100]' \
    --input-shapes   'question:[16, 20]' 'features:[16, 100, 2048]' 'spatials:[16, 100, 5]' 'segment_ids:[16, 20]' 'input_mask:[16, 20]' 'image_mask:[16, 100]' \
    > tmp_info/result-run-FP16_${outnodes}.txt

done