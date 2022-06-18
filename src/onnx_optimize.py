# -*- coding: utf-8 -*-

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


sourceOnnx = "/TRT2022_VilBERT/models/vilbert_model_vision_logit.onnx"
onnxSurgeonFile = "/TRT2022_VilBERT/models/vilbert_model_vision_logit_layernorm.onnx"


def reBuildGraph():
    bLayerNormPlugin = True
    nLayerNormPlugin = 0

    graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

    if bLayerNormPlugin:
        for node in graph.nodes:
            if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

                inputTensor = node.inputs[0]
                lastDivNode = node.o().o(0).o().o().o().o()

                layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), 
                                     inputs=[inputTensor],
                                     outputs=[lastDivNode.outputs[0]],
                                     )
                # attrs={"epsilon": node.o().o(0).o().o().i(1).attrs['value'].values.reshape(1)}
                graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1

                lastDivNode.outputs = []
                continue
            
        graph.cleanup()

        
    onnx.save(gs.export_onnx(graph), onnxSurgeonFile)

    print("finish encoder onnx-graphsurgeon!")
    print("%4d LayerNormPlugin" %nLayerNormPlugin)

def onnxOptimizer():
    original_model = onnx.load(onnxSurgeonFile)
    all_passes = onnxoptimizer.get_available_passes()
    passes = ['fuse_add_bias_into_conv','fuse_consecutive_concats','fuse_consecutive_squeezes', 'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv', 'fuse_transpose_into_gemm']
    optimized_model = onnxoptimizer.optimize(original_model, passes)
    onnx.save(optimized_model, onnxSurgeonFile)

if __name__ == "__main__":
    reBuildGraph()
    # onnxOptimizer()

