
import time
import numpy as np

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
        

save_input_features_batch = torch.load('/TRT2022_VilBERT/infer_batch_inputs/save_input_features_with_model_res')
for input_batch in save_input_features_batch:

    features = input_batch.features
    spatials = input_batch.spatials
    image_mask = input_batch.image_mask
    question = input_batch.question
    target = input_batch.target
    input_mask = input_batch.input_mask
    segment_ids = input_batch.segment_ids
    co_attention_mask = input_batch.co_attention_mask
    question_id = input_batch.question_id
    batch_size = input_batch.batch_size
    print('='*50)
    print('batch size: ', batch_size)
    break
    
    
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
    
import onnx
import onnxruntime


onxx_model_path = '/TRT2022_VilBERT/models/vilbert_model_vision_logit.onnx'
options = onnxruntime.SessionOptions()
# options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session = onnxruntime.InferenceSession(onxx_model_path, options, providers=['CUDAExecutionProvider'])
print('load onnx done')

input_name = session.get_inputs()
output_name = session.get_outputs() # [0].name
print('input_name: ')
for n in input_name:
    print(n.name, n.shape)
    
print('output_name: ')
for n in output_name:
    print(n.name, n.shape)


with open('/TRT2022_VilBERT/scores/vilbert_onnxruntime_infer_time.txt', 'w') as fw:
    for input_batch in save_input_features_batch:

        features = input_batch.features
        spatials = input_batch.spatials
        image_mask = input_batch.image_mask
        question = input_batch.question
        target = input_batch.target
        input_mask = input_batch.input_mask
        segment_ids = input_batch.segment_ids
        co_attention_mask = input_batch.co_attention_mask
        question_id = input_batch.question_id
        batch_size = input_batch.batch_size
        vision_logit = input_batch.vision_logit
        batch_loss = input_batch.batch_loss
        batch_score = input_batch.batch_score
        
        inputs = {'question': question.numpy(),
                'features': features.numpy(),
                'spatials': spatials.numpy(),
                'segment_ids': segment_ids.numpy(),
                'input_mask': input_mask.numpy(),
                'image_mask': image_mask.numpy(),
                }
        print('='*50)
        print('batch size: ', batch_size)

        onnx_vision_logit = session.run(['vision_logit'], inputs)[0]
        
        vision_logit = vision_logit.cpu().numpy()
        
        check(onnx_vision_logit, vision_logit, True)

        onnx_vision_logit = torch.from_numpy(onnx_vision_logit)

        onnx_batch_loss, onnx_batch_score = eval_logit(onnx_vision_logit, target)
        
        t0 = time.time()
        for i in range(30):
            _ = session.run(['vision_logit'], inputs)
        t1 = time.time()
        timePerInference = (t1 - t0) * 1000 / 30
        
        fw.write('='*50 + '\n')
        fw.write('batch_size: {},\ttimePerInference: {:.4f},\tbatch_loss: {:.4f},\tbatch_score: {:.4f}\n'.format(\
                    batch_size, timePerInference, onnx_batch_loss, onnx_batch_score))
        








