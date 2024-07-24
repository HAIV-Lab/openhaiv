/**
 * @file linear.cu
 * @brief Linear function implementation.
 * @version 0.0.2
 * @date 2024-01-15
 */

#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include "funcs.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCKSIZE 32


/**
 * Linear CUDA kernel.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param bias The bias tensor of shape (output_size).
 * @param output The output tensor of shape (batch_size, output_size).
 * @param batch_size The batch size.
 * @param input_size The input size.
 * @param output_size The output size.
 */
template <typename scalar_t>
__global__ void linear_cuda_kernel_1(
    const scalar_t * __restrict__ input,
    const scalar_t * __restrict__ weight,
    const scalar_t * __restrict__ bias,
    scalar_t * __restrict__ output,
    const int batch_size,
    const int input_size,
    const int output_size)
{
    // Get the index of the current thread
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is valid
    if (index < batch_size * output_size){
        // Get the batch index
        auto batch = index / output_size;

        // Get the output index
        auto output_index = index % output_size;

        // Compute the output
        auto output_value = bias[output_index];
        for (auto input_index = 0; input_index < input_size; input_index++){
            output_value += input[batch * input_size + input_index] * weight[output_index * input_size + input_index];
        }

        // Store the output
        output[index] = output_value;
    }
}


/**
 * Linear CUDA kernel.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param bias The bias tensor of shape (output_size).
 * @param output The output tensor of shape (batch_size, output_size).
 * @param batch_size The batch size, input.size(0).
 * @param input_size The input size, input.size(1) and weight.size(1).
 * @param output_size The output size, weight.size(0).
 */
template <typename scalar_t>
__global__ void linear_cuda_kernel_2(
    const scalar_t * __restrict__ input,
    const scalar_t * __restrict__ weight,
    const scalar_t * __restrict__ bias,
    scalar_t * __restrict__ output,
    const int batch_size,
    const int input_size,
    const int output_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t s_A[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t s_B[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t s_bias[BLOCKSIZE];

    scalar_t tmp = 0;
    #pragma unroll
    for (int i = 0; i < input_size; i += BLOCKSIZE) {
        if (row < batch_size && i + threadIdx.x < input_size)
            s_A[threadIdx.y][threadIdx.x] = input[row * input_size + i + threadIdx.x];
        if (col < output_size && i + threadIdx.y < input_size)
            s_B[threadIdx.y][threadIdx.x] = weight[col * input_size + i + threadIdx.y];
        if (i == 0 && col < output_size && threadIdx.y == 0)
            s_bias[threadIdx.x] = bias[col];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BLOCKSIZE; j++) {
            tmp += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < batch_size && col < output_size) {
        output[row * output_size + col] = tmp + s_bias[threadIdx.x];
    }
}


/**
 * Linear CUDA function.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param bias The bias tensor of shape (output_size).
 * @param mode The mode of cuda kernel.
 */
torch::Tensor linear_cuda(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const torch::Tensor & bias,
    const int mode)
{
    // Check tensors
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    // Get dimensions
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);

    // Allocate output tensor
    auto output = torch::zeros({batch_size, output_size}, input.options());

    // Compute output
    if (mode == 0) {
        auto threads = 1024;
        auto blocks = (batch_size * output_size + threads - 1) / threads;
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "linear_cuda", ([&] {
            linear_cuda_kernel_1<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                input_size,
                output_size);
        }));
    } else if (mode == 1) {
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
        dim3 blocks((output_size + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "linear_cuda", ([&] {
            linear_cuda_kernel_2<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                input_size,
                output_size);
        }));
    }

    return output;
}


/**
 * Linear function.
 *
 * @param input The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 * @param mode The mode of cuda kernel.
 * @return The output tensor, defined as:
 *         output = input * weight^T + bias
 */
torch::Tensor linear(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const c10::optional<torch::Tensor> & bias,
    const int mode)
{
    // Parse bias
    torch::Tensor _bias;
    if (bias.has_value()){
        _bias = bias.value();
    } else {
        _bias = torch::zeros({weight.size(0)}, weight.options());
    }

    // Compute output
    if (input.device().is_cuda() && weight.device().is_cuda() 
            && _bias.device().is_cuda()) {
        return linear_cuda(input, weight, _bias, mode);
    } else {
        return torch::mm(input, weight.t()) + _bias;
    }
}
