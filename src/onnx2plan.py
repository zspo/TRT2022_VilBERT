#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
from collections import OrderedDict
from copy import deepcopy
from glob import glob
import ctypes
import numpy as np
import onnx
# import onnxoptimizer
import onnx_graphsurgeon as gs
import tensorrt as trt


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--onnxFile", default="vilbert_model_vision_logit.onnx", type=str)
parser.add_argument("--trtFile", default="vilbert_model_vision_logit_fp16.plan", type=str)
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()


models_path = "/TRT2022_VilBERT/models/"

sourceOnnx = os.path.join(models_path, args.onnxFile)
# onnxSurgeonFile = os.path.join(models_path, args.onnxSurgeonFile)
planFilePath = "/TRT2022_VilBERT/libs/"
soFileList = glob(planFilePath + "*.so")
print(soFileList)
trtFile = os.path.join(models_path, args.trtFile)


def run():

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    if len(soFileList) > 0:
        print("Find Plugin %s!"%soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 12 << 30
    if args.fp16:
        print('[FP16]')
        config.flags = 1 << int(trt.BuilderFlag.FP16)
    #     config.flags = config.flags | 1<<int(trt.BuilderFlag.STRICT_TYPES)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(sourceOnnx):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(sourceOnnx, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing onnx file!")
        
    # for i in range(network.num_layers):
    #     if network.get_layer(i).type == trt.LayerType.FULLY_CONNECTED or network.get_layer(i).type == trt.LayerType.MATRIX_MULTIPLY or network.get_layer(i).type == trt.LayerType.SOFTMAX:
    #         print(network.get_layer(i).type)
    #         network.get_layer(i).precision = trt.DataType.FLOAT
    #         network.get_layer(i).set_output_type(0, trt.DataType.FLOAT)
    

    profile.set_shape('question', [1, 20], [16, 20], [256, 20])
    profile.set_shape('features', [1, 100, 2048], [16, 100, 2048], [256, 100, 2048])
    profile.set_shape('spatials', [1, 100, 5], [16, 100, 5], [256, 100, 5])
    profile.set_shape('segment_ids', [1, 20], [16, 20], [256, 20])
    profile.set_shape('input_mask', [1, 20], [16, 20], [256, 20])
    profile.set_shape('image_mask', [1, 100], [16, 100], [256, 100])

    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)

if __name__ == "__main__":
    run()
