
import time
import numpy as np

import torch
import torch.nn as nn


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
        

save_input_features_batch = torch.load('cache_input_features/features_with_logit_res_batch')
for batch in save_input_features_batch:
    print('features\n', batch.features.shape)
    print('spatials\n', batch.spatials.shape)
    print('image_mask\n', batch.image_mask.shape)
    print('question\n', batch.question.shape)
    print('target\n', batch.target.shape)
    print('input_mask\n', batch.input_mask.shape)
    print('segment_ids\n', batch.segment_ids.shape)
    print('co_attention_mask\n', batch.co_attention_mask.shape)
    print('vision_logit\n', batch.vision_logit.shape)
    print('='*50)
#     break
    
    
def check(a, b, weak=False, info=""):  # 用于比较 TF 和 TRT 的输出结果
    epsilon = 1e-6
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:" % info, res, diff0, diff1)
    
    
import onnx
import onnxruntime


onxx_model_path = r'./vilbert_model_v-logit.onnx'
options = onnxruntime.SessionOptions()
# options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session = onnxruntime.InferenceSession(onxx_model_path, options)
print('load onnx done')

input_name = session.get_inputs()
output_name = session.get_outputs() # [0].name
print('input_name: ')
for n in input_name:
    print(n.name)
    
print('output_name: ')
for n in output_name:
    print(n.name)
    print(n.shape)

    
loss_func = nn.BCEWithLogitsLoss(reduction='mean')
result = []
eval_start_time = time.time()
for batch in save_input_features_batch:
    
    inputs = {'question':  batch.question.numpy(),
              'features': batch.features.numpy(),
              'spatials': batch.spatials.numpy(),
              'segment_ids': batch.segment_ids.numpy(),
              'input_mask': batch.input_mask.numpy(),
              'image_mask': batch.image_mask.numpy(),
#               'co_attention_mask': batch.co_attention_mask.numpy()
             }
    print('='*50)
    print('batch size: ', batch.question.shape[0])

    # (batch.question,batch.features,batch.spatials,batch.segment_ids,batch.input_mask,batch.image_mask,batch.co_attention_mask)
    vision_logit = session.run(['4240'], inputs)[0]
    
    print(vision_logit.shape)
    
    torch_output_vision_logit = batch.vision_logit.numpy()
    
    check(torch_output_vision_logit, vision_logit, True)
    
    
    
#     vision_logit = torch.from_numpy(vision_logit)
#     target = batch.target
#     loss = loss_func(vision_logit, target)
#     loss = loss.mean() * target.size(1)
#     _, select_idx = torch.max(vision_logit, dim=1)
#     select_target = target.squeeze(2).gather(1, select_idx.view(-1,1))
#     batch_score = torch.sum(select_target>0.5).item()
#     print(float(loss), float(batch_score))
    
#     result.append(res[0])
#     break
# result = np.concatenate(result)
# eval_end_time = time.time()
# eval_duration_time = eval_end_time - eval_start_time
# print()
# print("Evaluate total time (seconds): {0:.4f}".format(eval_duration_time))
# print("Evaluate avg time (seconds): {0:.4f}".format(eval_duration_time / len(save_input_features_batch_8)))




