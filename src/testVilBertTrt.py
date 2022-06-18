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
import torch.nn as nn


class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self, features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, batch_size, vision_logit=None, loss=None, batch_score=None):
        self.features           = features
        self.spatials           = spatials
        self.image_mask         = image_mask
        self.question           = question
        self.target             = target
        self.input_mask         = input_mask
        self.segment_ids        = segment_ids
        self.co_attention_mask  = co_attention_mask
        self.question_id        = question_id
        self.batch_size         = batch_size
        self.vision_logit       = vision_logit
        self.batch_loss         = loss
        self.batch_score        = batch_score


planFilePath = "./"
soFileList = glob(planFilePath + "*.so")
print(soFileList)
trtFile = "/TRT2022_VilBERT/models/vilbert_model_vision_logit.plan"

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


def eval_logit(vision_logit, target):
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    loss = loss_func(vision_logit, target)
    loss = loss.mean() * target.size(1)
    _, select_idx = torch.max(vision_logit, dim=1)
    select_target = target.squeeze(2).gather(1, select_idx.view(-1,1))
    batch_score = torch.sum(select_target>0.5).item()
    return float(loss), float(batch_score)


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

    with open('/TRT2022_VilBERT/scores/vilbert_tensorrt_infer_time.txt', 'w') as fw:
        save_input_features_batch = torch.load('/TRT2022_VilBERT/infer_batch_inputs/save_input_features_with_model_res')
        for input_batch in save_input_features_batch:

            features = input_batch.features.numpy()
            spatials = input_batch.spatials.numpy()
            image_mask = input_batch.image_mask.numpy()
            question = input_batch.question.numpy()
            target = input_batch.target
            input_mask = input_batch.input_mask.numpy()
            segment_ids = input_batch.segment_ids.numpy()
            batch_size = input_batch.batch_size
            vision_logit = input_batch.vision_logit
            batch_loss = input_batch.batch_loss
            batch_score = input_batch.batch_score

            for index, inputs in enumerate([question, features, spatials, segment_ids, input_mask, image_mask]):
                context.set_binding_shape(index, inputs.shape)

            bufferH = []
            bufferH.append(question.astype(np.int32).reshape(-1))
            bufferH.append(features.astype(np.float32).reshape(-1))
            bufferH.append(spatials.astype(np.float32).reshape(-1))
            bufferH.append(segment_ids.astype(np.int32).reshape(-1))
            bufferH.append(input_mask.astype(np.int32).reshape(-1))
            bufferH.append(image_mask.astype(np.int32).reshape(-1))

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

            indexOut = engine.get_binding_index('vision_logit')
            trt_vision_logit = bufferH[indexOut]
            print(trt_vision_logit.shape)

            vision_logit = vision_logit.cpu().numpy()
            check(trt_vision_logit, vision_logit, True)

            trt_vision_logit = torch.from_numpy(trt_vision_logit)
            trt_batch_loss, trt_batch_score = eval_logit(trt_vision_logit, target)
            
            t0 = time_ns()
            for i in range(30):
                context.execute_v2(bufferD)
            t1 = time_ns()
            timePerInference = (t1 - t0)/1000/1000/30
            
            fw.write('='*50 + '\n')
            fw.write('batch_size: {},\ttimePerInference: {:.4f},\tbatch_loss: {:.4f},\tbatch_score: {:.4f}\n'.format(\
                        batch_size, timePerInference, trt_batch_loss, trt_batch_score))

            for i in range(nInput + nOutput):
                cudart.cudaFree(bufferD[i])


if __name__ == '__main__':
    run()