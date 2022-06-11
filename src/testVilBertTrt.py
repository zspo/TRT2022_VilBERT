#!/usr/bin/python

import os
import sys
import ctypes
import numpy as np
from glob import glob
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
import torch


class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self,features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id):
        self.features   = features
        self.spatials  = spatials
        self.image_mask = image_mask
        self.question    = question
        self.target   = target
        self.input_mask   = input_mask
        self.segment_ids   = segment_ids
        self.co_attention_mask   = co_attention_mask
        self.question_id   = question_id
        self.vision_logit = vision_logit


sourceOnnx = f"./vilbert_model_v-logit.onnx"
onnxSurgeonFile = f"./vilbert_model_v-logit.onnx_surgeon.onnx"
planFilePath = "./"
soFileList = glob(planFilePath + "*.so")
print(soFileList)
trtFile = f"./vilbert_model_v-logit.onnx.plan"
trtScoreFile = "./trtScore.txt"
input_features = './features_with_logit_res_batch'

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)


def check(a, b, weak=False, info=""):  # 用于比较 TF 和 TRT 的输出结果
    epsilon = 1e-6
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:" % info, res, diff0, diff1)


def run():
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as encoderF:
            engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
        if engine is None:
            print("Failed loading %s"%trtFile)
            return
        print("Succeeded loading %s"%trtFile)
    else:
        print("Failed finding %s"%trtFile)
        return

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    context = engine.create_execution_context()

    save_input_features_batch = torch.load(input_features)
    for batch in save_input_features_batch:
        question = batch.question.numpy()
        features = batch.features.numpy()
        spatials = batch.spatials.numpy()
        segment_ids = batch.segment_ids.numpy()
        input_mask = batch.input_mask.numpy()
        image_mask = batch.image_mask.numpy()

        batchSize = question.shape[0]

        context.set_binding_shape(0, question.shape)
        context.set_binding_shape(1, features.shape)
        context.set_binding_shape(2, spatials.shape)
        context.set_binding_shape(3, segment_ids.shape)
        context.set_binding_shape(4, input_mask.shape)
        context.set_binding_shape(5, image_mask.shape)
        #for i in range(nInput + nOutput):
        #    print("Input ->" if engine.binding_is_input(i) else "Output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_dtype(i), engine.get_binding_name(i))
        #print("Finish all input binding: %s"%context.all_binding_shapes_specified)

        bufferH = []
        bufferH.append( question.astype(np.int32).reshape(-1) )
        bufferH.append( features.astype(np.float32).reshape(-1) )
        bufferH.append( spatials.astype(np.float32).reshape(-1) )
        bufferH.append( segment_ids.astype(np.int32).reshape(-1) )
        bufferH.append( input_mask.astype(np.int32).reshape(-1) )
        bufferH.append( image_mask.astype(np.int32).reshape(-1) )

        for i in range(nInput, nInput + nOutput):
            bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

        bufferD = []
        for i in range(nInput + nOutput):
            bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # warm up
        for i in range(10):
            context.execute_v2(bufferD)

        # test infernece time
        t0 = time_ns()
        for i in range(30):
            context.execute_v2(bufferD)
        t1 = time_ns()
        timePerInference = (t1-t0)/1000/1000/30

        indexOut = engine.get_binding_index('4238')
        print(bufferH[indexOut].shape)

        # indexOut = engine.get_binding_index('vision_logit')
        # print(bufferH[indexOut].shape)

        torch_output_vision_logit = batch.vision_logit.numpy()
        check(torch_output_vision_logit, bufferH[indexOut])

        print(f'{batchSize}\t{timePerInference}\t')

        for i in range(nInput + nOutput):
            cudart.cudaFree(bufferD[i])


if __name__ == '__main__':
    run()