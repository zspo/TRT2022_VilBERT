#include <cuda.h>
#include "cublas_v2.h"
#include <cub/cub.cuh>
#include "cuda_fp16.h"

#include "LayerNormPlugin.h"
#include "common.cuh"

#include <numeric>


using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;


inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename T, int TPB, int VPT>
__global__ void skipln_vec(
    const int ld, const T* input, T* output)
{
    float epsilon {1e-5};
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T in_local[VPT];
    // T gamma_local[VPT];
    copy<sizeof(T) * VPT>(&input[idx], in_local);

    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const T tmp = rld * in_local[it];
        local += tmp;
        local2 += tmp * in_local[it];
    }

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<T>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + (T)epsilon);
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] = (in_local[it] - mu) * rsigma;
    }
    /* */

    copy<sizeof(T) * VPT>(in_local, &output[idx]);
}

template <typename T, unsigned TPB>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, T* output)
{

    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);
    const int idx = offset + threadIdx.x;
    T val = 0;

    if (threadIdx.x < ld)
    {

        val = input[idx];

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    layerNormSmall<T, T, TPB>(val, threadData, ld, idx, output);
}

template <typename T, unsigned TPB>
__global__ void skipLayerNormKernel(
    const int ld, const T* input, T* output)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        T val = input[idx];

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, T, TPB>(threadData, ld, offset, output);
}


int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    // int nBlock = inputDesc[0].dims.d[0], nValuePerBlock = 1;
    // for (int i = 1; i < inputDesc[0].dims.nbDims; ++i)
    // {
    //     nValuePerBlock *= inputDesc[0].dims.d[i];
    // }
    
    int dim = inputDesc[0].dims.nbDims;
    const int n = volume(inputDesc[0].dims);
    // const int n = 60;
    // printf("n: %d\n", n);
    // printf("dim: %d\n", dim);
    // for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    // {
    //    printf("dim i: %d, value: %d\n", i, inputDesc[0].dims.d[i]);
    // }

    int ld = inputDesc[0].dims.d[dim - 1];
    // int ld = 5;
    // printf("ld: %d\n", ld);

    if (inputDesc[0].type == DataType::kFLOAT)
    {  
        // printf("input float\n");
        const auto* const input = static_cast<const float*>(inputs[0]);
        auto* output = static_cast<float*>(outputs[0]);

        assert(n % ld == 0);
        const int gridSize = n / ld;
        constexpr int VPT = 16 / sizeof(float);
        if (ld <= 32)
        {
            constexpr int blockSize = 32;
            skipLayerNormKernelSmall<float, blockSize>
                <<<gridSize, blockSize, 0, stream>>>(ld, input, output);
        }
        else if (ld == 768)
        {
            constexpr int TPB = 768 / VPT;
            skipln_vec<float, TPB, VPT><<<gridSize, TPB, 0, stream>>>(ld, input, output);
        }
        else if (ld == 1024)
        {
            constexpr int TPB = 1024 / VPT;
            skipln_vec<float, TPB, VPT><<<gridSize, TPB, 0, stream>>>(ld, input, output);
        }
        else
        {
            constexpr int blockSize = 256;
            skipLayerNormKernel<float, blockSize>
                <<<gridSize, blockSize, 0, stream>>>(ld, input, output);
        }
    }
    else 
    {   
        // printf("input __half\n");
        const auto* const input = static_cast<const __half*>(inputs[0]);
        auto* output = static_cast<__half*>(outputs[0]);

        assert(n % ld == 0);
        const int gridSize = n / ld;
        constexpr int VPT = 16 / sizeof(__half);
        if (ld <= 32)
        {
            constexpr int blockSize = 32;
            skipLayerNormKernelSmall<__half, blockSize>
                <<<gridSize, blockSize, 0, stream>>>(ld, input, output);
        }
        else if (ld == 768)
        {
            constexpr int TPB = 768 / VPT;
            skipln_vec<__half, TPB, VPT><<<gridSize, TPB, 0, stream>>>(ld, input, output);
        }
        else if (ld == 1024)
        {
            constexpr int TPB = 1024 / VPT;
            skipln_vec<__half, TPB, VPT><<<gridSize, TPB, 0, stream>>>(ld, input, output);
        }
        else
        {
            constexpr int blockSize = 256;
            skipLayerNormKernel<__half, blockSize>
                <<<gridSize, blockSize, 0, stream>>>(ld, input, output);
        }
        // __half* outputres = (__half*)output;
    }
    
    // cudaDeviceSynchronize();
    
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

