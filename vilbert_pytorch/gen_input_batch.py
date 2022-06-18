import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict
import sys
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from vilbert.task_utils import LoadDatasetEval, LoadLosses, ForwardModelsTrain, ForwardModelsVal, EvaluatingModel

import vilbert.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.", 
    )
    parser.add_argument(
        "--batch_size", default=1000, type=int, help="what is the batch size?"
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--split", default="", type=str, help="which split to use."
    )

    args = parser.parse_args()
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BaseBertForVLTasks     
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)

    # timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0]
    timeStamp = args.from_pretrained.split('/')[1] + '-' + args.save_name
    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)
    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # torch.distributed.init_process_group(backend="nccl")
    
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )
    
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    num_labels = 1
    model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=num_labels, default_gpu=default_gpu
            )
    model.to(device)
    model.eval()
    task_losses = LoadLosses(args, task_cfg, args.tasks.split('-'))


    use_raw_data = False

    if use_raw_data:
        save_input_features = []
        for batch_size in [8, 16, 32, 64, 128, 256]:
            print(batch_size)
            args.batch_size = batch_size
            print(args.batch_size)
            logger.info('load datasets')
            task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val \
                                = LoadDatasetEval(args, task_cfg, args.tasks.split('-'))
            
            task_id = task_ids[0]
            print('task_id: ', task_id)

            for i, batch in enumerate(task_dataloader_val[task_id]):
                features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = batch

                results, others = [], []
                loss, score, batch_size, results, others, vision_logit = EvaluatingModel(args, task_cfg, device, \
                        task_id, batch, model, task_losses, results, others)

                tmp_feature = InputFeature(features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, batch_size, vision_logit, loss, score)

                print(f'batch_size: {batch_size}\t, loss: {loss}\t, score: {score}\t')
                break
                
            save_input_features.append(tmp_feature)

        torch.save(save_input_features, '/TRT2022_VilBERT/infer_batch_inputs/save_input_features_batch_all')
        print('save done')
    else:
        save_input_features_with_model_res = []
        save_input_features_batch = torch.load('/TRT2022_VilBERT/infer_batch_inputs/save_input_features_batch_all')
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

            task_id = 'TASK4'
            batch = (features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id)
            results, others = [], []
            loss, score, batch_size, results, others, vision_logit = EvaluatingModel(args, task_cfg, device, \
                    task_id, batch, model, task_losses, results, others)

            tmp_feature = InputFeature(features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, batch_size, vision_logit, loss, score)

            print(f'batch_size: {batch_size},\t loss: {loss},\t score: {score}')

            save_input_features_with_model_res.append(tmp_feature)
        torch.save(save_input_features_with_model_res, '/TRT2022_VilBERT/infer_batch_inputs/save_input_features_with_model_res')
        print('save done')




if __name__ == "__main__":

    main()
