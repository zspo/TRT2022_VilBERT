# TRT2022_VilBERT

本项目是基于ViLBERT进行TensorRT的部署。ViLBERT模型是为视觉-语言任务训练非任务专用的视觉语言表征的BERT融合模型，可以学习视觉内容和文本内容的与特定任务无关的联合表征。

原始论文：[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)

原始模型参考：https://github.com/jiasenlu/vilbert_beta


### 模型简介：
主要针对RefCOCO+任务进行评测

### 运行环境：

### 项目结构：

```shell
.
├── README.md
├── models
│   └── readme.md                   # onnx 及 trt plan存放路径
├── plugins
│   └── LayerNormPlugin             # trt plugin路径
├── scores                          # 评测结果存放路径
├── src
│   ├── onnx2plan.py                # onnx转plan
│   ├── onnx_optimize.py            # onnx图优化
│   └── testVilBertTrt.py           # trt推理评测
└── vilbert_pytorch
    ├── README.md
    ├── cache_input_features        # 预生成的Batch评测数据路径
    ├── config                      # 模型配置
    ├── convert_onnx.py             # torch模型转onnx
    ├── data                        # 原始数据
    ├── eval_retrieval.py
    ├── eval_tasks.py
    ├── fig
    ├── gen_input_batch.py          # 产出Batch评测数据
    ├── onnx_test.py                # onnx推理评测
    ├── requirements.txt
    ├── save                        # torch模型所需的预训练模型
    ├── script
    ├── tools
    ├── torch_model_test.py         # torch模型推理评测
    ├── train_baseline.py
    ├── train_concap.py
    ├── train_tasks.py
    ├── vilbert
    └── vlbert_tasks.yml
```


### 部署优化流程：

##### 1. 参考vilber_pytorch准备预训练模型及RefCOCO+所需数据
##### 2. 准备评测所需Batch数据
```shell
cd vilbert_pytorch
python eval_gen_input_batch.py --bert_model bert-base-uncased --from_pretrained save/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 4 --batch_size 8
```
##### 3. Pytorch转Onnx
```shell
cd vilbert_pytorch
python convert_onnnx.py --bert_model bert-base-uncased --from_pretrained save/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 4 --batch_size 8
```
##### 4. Pytorch推理评测
```shell
cd vilbert_pytorch
python torch_model_test.py --bert_model bert-base-uncased --from_pretrained save/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 4 --batch_size 8
```

##### 5. Onnx图优化

##### 6. 编译Plugin算子

##### 7. Onnx转Plan
```
cd src
python onnx2plan.py
```

##### 8. Trt推理评测
```
cd src
python testVilBertTrt.py
```


### 性能对比：

##### Reference：