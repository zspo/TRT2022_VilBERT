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
├── infer_batch_inputs              # 推理测试所需的batch输入及torch模型的target、logit、batch_loss、batch_score
├── models
│   └── readme.md                   # onnx 及 trt plan存放路径
├── plugins
│   └── LayerNormPlugin             # trt plugin路径
├── scores                          # 评测结果存放路径
├── script                          # 运行shell脚本
├── src
│   ├── onnx2plan.py                # onnx转plan
│   ├── onnx_optimize.py            # onnx图优化
│   └── testVilBertTrt.py           # tensorrt推理评测
└── vilbert_pytorch
    ├── README.md
    ├── config                      # 模型配置
    ├── convert_onnx.py             # torch模型转onnx
    ├── data                        # 原始数据
    ├── eval_tasks.py
    ├── gen_input_batch.py          # 产出Batch评测数据
    ├── onnx_test.py                # onnx推理评测
    ├── requirements.txt
    ├── save                        # torch模型所需的预训练模型
    ├── script
    ├── tools
    ├── torch_model_test.py         # torch模型推理评测
    ├── vilbert
    └── vlbert_tasks.yml
```


### 运行流程：

##### 1. 参考vilber_pytorch准备预训练模型及RefCOCO+所需数据

(在参考项目的基础上添加以下脚本及修改部分脚本简化流程)
##### 2. 准备评测所需Batch数据
```shell
sh script/gen_inputs.sh
```
##### 3. Pytorch推理评测
```shell
sh script/torch_infer.sh
```
##### 4. Pytorch转Onnx
```shell
sh script/torch2onnx.sh
```
##### 5. OnnxRuntime推理评测
```shell
sh script/onnxruntime_infer.sh
```
##### 6. Onnx图优化

##### 7. 编译Plugin算子

##### 8. Onnx转Plan
```shell
sh script/onnx2plan.sh
```

##### 9. TensortRT推理评测
```shell
sh script/tensorrt_infer.sh
```

### 优化细节：

### 性能对比：

##### 评测说明：


| BatchSize | Pytorch(C) | Pytorch(G) | OnnxRuntime(G) | TensorRT | TensorRT(Opt) |
| --------- | ---------  | ---------  | -------------  | -------  | ------------  |
| 8         | 417.57     | 25.33      | 11.31          | 8.70     |               |
| 16        | 984.57     | 31.85      | 21.35          | 16.44    |               |
| 32        | 1508.12    | 55.17      | 38.42          | 28.47    |               |
| 64        | 2553.40    | 106.70     | 75.61          | 54.88    |               |
| 128       | **         | 214.04     | 149.27         | 106.12   |               |
| 256       | **         | 428.90     | 299.14         | 213.98   |               |


### Reference：