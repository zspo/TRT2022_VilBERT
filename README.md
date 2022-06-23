# TRT2022_VilBERT

本项目是基于ViLBERT进行TensorRT的部署。ViLBERT模型是为视觉-语言任务训练非任务专用的视觉语言表征的BERT融合模型，可以学习视觉内容和文本内容的与特定任务无关的联合表征。

原始论文：[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)

原始模型参考：https://github.com/jiasenlu/vilbert_beta


### 模型简介：
主要针对RefCOCO+任务进行评测

### 运行环境：

##### 硬件环境
+ GPU: NVIDIA A10
+ Driver Version: 510.73.08
+ CUDA: 11.6
+ Docker：registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
+ TensorRT: 8.4.1.5

##### 主要Python环境
+ onnx==1.11.0
+ onnx-graphsurgeon==0.3.19
+ onnxruntime-gpu==1.11.1
+ pytorch-pretrained-bert==0.6.2
+ torch==1.8.1+cu111


### 项目结构：

```shell
.
├── README.md
├── requirements.txt
├── fig
├── infer_batch_inputs              # 推理测试所需的batch输入及torch模型的target、logit、batch_loss、batch_score
├── libs
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

##### 环境准备
```shell
1. docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
2. nvidia-docker run -it --name trt2022_trt84 --privileged=true -v /TRT2022_VilBERT/:/TRT2022_VilBERT/ registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
3. cd /TRT2022_VilBERT && pip install -r requirements.txt
```

##### 1. 参考vilbert_pytorch准备预训练模型及RefCOCO+所需数据

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
```shell
sh script/onnx_optimize.sh
```
##### 7. 编译Plugin算子
```shell
cd plugins/LayerNormPlugin
make clean
make
cp LayerNormPlugin.so /TRT2022_VilBERT/libs
```
##### 8. Onnx转Plan
```shell
sh script/onnx2plan.sh
```
##### 9. TensortRT推理评测
```shell
sh script/tensorrt_infer.sh
```

### 优化细节：

##### 原始模型修改以生成合适的Onnx

##### Onnx导出版本及Pytorch版本导致节点生成不同

##### 使用polygraphy进行FP16精度下的层输出对比

### 性能对比：

##### 评测说明：

用同一份输入评测数据：`infer_batch_inputs/save_input_features_with_model_res`
首先进行模型的warm up，再进行模型的30次推理，时间取平均

+ Pytorch(C):         原始Vilbert模型在CPU上运行
+ Pytorch(G):         原始Vilbert模型在A10-GPU上运行
+ PytorchV2(G):       优化LayerNorm，用torch.nn.LayerNorm替代自定义BertLayerNorm后的Vilbert模型在A10-GPU上运行
+ OnnxRuntime(G)：    优化后的Vilbert模型在A10-GPU上以OnnxRuntime进行推理
+ TensorRT:           基于Onnx转的Plan模型以TensorRT进行推理
+ TensorRT(FP16):     
+ TensorRT(Opt):      基于优化后的Onnx及优化后的TensorRT进行推理


| BatchSize | Pytorch(C) | Pytorch(G) | PytorchV2(G) | OnnxRuntime(G) | TensorRT | TensorRT(Opt) |
| --------- | ---------  | ---------  | ---------    | -------------  | -------  | ------------  |
| 8         | 417.57     | 25.33      | 19.41        | 11.31          | 8.70     |               |
| 16        | 984.57     | 31.85      | 26.15        | 21.35          | 16.44    |               |
| 32        | 1508.12    | 55.17      | 46.07        | 38.42          | 28.47    |               |
| 64        | 2553.40    | 106.70     | 88.92        | 75.61          | 54.88    |               |
| 128       | **         | 214.04     | 178.74       | 149.27         | 106.12   |               |
| 256       | **         | 428.90     | 360.32       | 299.14         | 213.98   |               |


### Reference：