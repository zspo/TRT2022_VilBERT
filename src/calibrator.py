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
import numpy as np
from cuda import cudart
import tensorrt as trt


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


class DataLoader:
    def __init__(self, batch_size):
        self.index = 0
        self.batch_size = batch_size
        self.length = 1

        self.calibration_data = = {'question': question.numpy(),
                                    'features': features.numpy(),
                                    'spatials': spatials.numpy(),
                                    'segment_ids': segment_ids.numpy(),
                                    'input_mask': input_mask.numpy(),
                                    'image_mask': image_mask.numpy(),
                                    }

    def reset(self):
        self.index = 0

    def next_batch(self):
            return np.ascontiguousarray(self.calibration_data)
        else:
            return np.array([])
            
    def __len__(self):
        return self.length

    

class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, stream, batchSize, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = stream
        self.batchSize = batchSize
        self.buffeSize = trt.volume(self.stream.) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.count = 0

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def get_batch_size(self):  # do NOT change name
        return self.batchSize

    def get_batch(self, inputNodeName=None):  # do NOT change name
        batch_data = self.stream.next_batch()
        cudart.cudaMemcpy(self.dIn, batch_data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        return [int(self.dIn)]
        else:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")


if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator(5, 16, "./int8.cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
