/**
 * @file quantlinear.cu
 * @brief Quantized linear function implementation.
 * @version 0.0.1
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
 * Quantized linear CUDA kernel.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param input_bits The number of bits of input tensor.
 * @param input_offset The offset of input tensor.
 * @param input_scale The scale of input tensor. 
 * @param input_zero The zero of input tensor.
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param weight_bits The number of bits of weight tensor.
 * @param weight_offset The offset of weight tensor.
 * @param weight_scale The scale of weight tensor.
 * @param weight_zero The zero of weight tensor.
 * @param bias The bias tensor of shape (output_size).
 * @param output The output tensor of shape (batch_size, output_size).
 * @param batch_size The batch size.
 * @param input_size The input size.
 * @param output_size The output size.
 */
template <typename input_t, typename weight_t>
__global__ void quantlinear_cuda_kernel(
    const unsigned char * __restrict__ input,
    const int input_bits,
    const unsigned char input_offset,
    const float * __restrict__ input_scale,
    const float * __restrict__ input_zero,
    const unsigned char * __restrict__ weight,
    const int weight_bits,
    const unsigned char weight_offset,
    const float * __restrict__ weight_scale,
    const float * __restrict__ weight_zero,
    const float * __restrict__ bias,
    float * __restrict__ output,
    const int batch_size,
    const int input_size,
    const int output_size)
{
    // Get the row and column
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory
    __shared__ unsigned char s_A[BLOCKSIZE][BLOCKSIZE];
    __shared__ unsigned char s_B[BLOCKSIZE][BLOCKSIZE];
    __shared__ float s_scale[BLOCKSIZE][BLOCKSIZE];
    __shared__ float s_A_zero[BLOCKSIZE];
    __shared__ float s_B_zero[BLOCKSIZE];
    __shared__ float s_bias[BLOCKSIZE];

    // Accumulator
    float tmp = 0.0f;

    // Loop over all tiles
    #pragma unroll
    for (int i = 0; i < input_size; i += BLOCKSIZE){
        // Load the tile
        if (row < batch_size && i + threadIdx.x < input_size) {
            int ele_idx = row * input_size + i + threadIdx.x;
            int byte_idx = ele_idx * input_bits / 8;
            int bit_idx = ele_idx * input_bits % 8;
            s_A[threadIdx.y][threadIdx.x] = (input[byte_idx] >> bit_idx) & ((1 << input_bits) - 1);
            if (bit_idx + input_bits > 8)
                s_A[threadIdx.y][threadIdx.x] |= (input[byte_idx + 1] << (8 - bit_idx)) & ((1 << input_bits) - 1);
        }
        
        if (col < output_size && i + threadIdx.y < input_size) {
            int ele_idx = col * input_size + i + threadIdx.y;
            int byte_idx = ele_idx * weight_bits / 8;
            int bit_idx = ele_idx * weight_bits % 8;
            s_B[threadIdx.y][threadIdx.x] = (weight[byte_idx] >> bit_idx) & ((1 << weight_bits) - 1);
            if (bit_idx + weight_bits > 8)
                s_B[threadIdx.y][threadIdx.x] |= (weight[byte_idx + 1] << (8 - bit_idx)) & ((1 << weight_bits) - 1);
        }

        // Load the scale and zero
        if (i == 0 && row < batch_size && col < output_size)
            s_scale[threadIdx.y][threadIdx.x] = input_scale[row] * weight_scale[col];

        if (i == 0 && row < batch_size && threadIdx.x == 0) {
            s_A_zero[threadIdx.y] = input_zero[row];
        }

        if (i == 0 && col < output_size && threadIdx.y == 0) {
            s_B_zero[threadIdx.x] = weight_zero[col];
            s_bias[threadIdx.x] = bias[col];
        }

        __syncthreads();

        // Loop over the values in the tile
        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++){
            // Compute the input value
            unsigned char input_ele = s_A[threadIdx.y][k] - input_offset;
            input_t _input_value = (input_t)input_ele;
            float input_value = (_input_value + s_A_zero[threadIdx.y]);

            // Compute the weight value
            unsigned char weight_ele = s_B[k][threadIdx.x] - weight_offset;
            weight_t _weight_value = (weight_t)weight_ele;
            float weight_value = (_weight_value + s_B_zero[threadIdx.x]);

            // Accumulate
            tmp += input_value * weight_value * s_scale[threadIdx.y][threadIdx.x];
        }

        __syncthreads();
    }

    // Store the output
    if (row < batch_size && col < output_size) {
        output[row * output_size + col] = tmp + s_bias[threadIdx.x];
    }
}


/**
 * Quantized linear function, running on CUDA.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param input_bits The number of bits of input tensor.
 * @param input_sign The sign of input tensor.
 * @param input_scale The scale of input tensor.
 * @param input_zero The zero of input tensor.
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param weight_bits The number of bits of weight tensor.
 * @param weight_sign The sign of weight tensor.
 * @param weight_scale The scale of weight tensor.
 * @param weight_zero The zero of weight tensor.
 * @param bias The bias tensor of shape (output_size).
 * @param batch_size The batch size, input.size(0).
 * @param input_size The input size, input.size(1) and weight.size(1).
 * @param output_size The output size, weight.size(0).
 */
torch::Tensor quantlinear_cuda(
    const torch::Tensor & input,
    const int input_bits,
    const int input_sign,
    const torch::Tensor & input_scale,
    const torch::Tensor & input_zero,
    const torch::Tensor & weight,
    const int weight_bits,
    const int weight_sign,
    const torch::Tensor & weight_scale,
    const torch::Tensor & weight_zero,
    const torch::Tensor & bias,
    const int batch_size,
    const int input_size,
    const int output_size)
{
    // Allocate the output tensor
    auto output = torch::zeros({batch_size, output_size}, input.options()).to(torch::kFloat32);

    // Get the offset
    unsigned char input_offset = 0;
    if (input_sign)
        input_offset = 1 << (input_bits - 1);

    unsigned char weight_offset = 0;
    if (weight_sign)
        weight_offset = 1 << (weight_bits - 1);

    // Get the threads and blocks
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((output_size + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);

    // Get the types
    torch::ScalarType input_type = input_sign ? torch::kInt8 : torch::kUInt8;
    torch::ScalarType weight_type = weight_sign ? torch::kInt8 : torch::kUInt8;
    
    // Call the CUDA kernel
    AT_DISPATCH_ALL_TYPES(input_type, "quantlinear_cuda_1", [&] {
        using input_t = scalar_t;
        AT_DISPATCH_ALL_TYPES(weight_type, "quantlinear_cuda_2", [&] {
            using weight_t = scalar_t;
            quantlinear_cuda_kernel<input_t, weight_t><<<blocks, threads>>>(
                input.data_ptr<unsigned char>(),
                input_bits,
                input_offset,
                input_scale.data_ptr<float>(),
                input_zero.data_ptr<float>(),
                weight.data_ptr<unsigned char>(),
                weight_bits,
                weight_offset,
                weight_scale.data_ptr<float>(),
                weight_zero.data_ptr<float>(),
                bias.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                input_size,
                output_size);
        });
    });
    
    return output;
}


/**
 * Quantized linear function.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param input_des The description of input tensor, which is a 1D tensor of size 4.
 * @param input_scale The scale of input tensor.
 * @param input_zero The zero of input tensor.
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param weight_des The description of weight tensor, which is a 1D tensor of size 4.
 * @param weight_scale The scale of weight tensor.
 * @param weight_zero The zero of weight tensor.
 * @param bias The bias tensor of shape (output_size).
 */
torch::Tensor quantlinear(
    const torch::Tensor & input,
    const torch::Tensor & input_des,
    const torch::Tensor & input_scale,
    const torch::Tensor & input_zero,
    const torch::Tensor & weight,
    const torch::Tensor & weight_des,
    const torch::Tensor & weight_scale,
    const torch::Tensor & weight_zero,
    const c10::optional<torch::Tensor> & bias)
{
    // Check the input
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(input_des);
    CHECK_INPUT(input_scale);
    CHECK_INPUT(input_zero);
    CHECK_INPUT(weight_des);
    CHECK_INPUT(weight_scale);
    CHECK_INPUT(weight_zero);

    // Parse descriptions
    const int input_bits = input_des[0].item<int>();
    const int input_sign = input_des[1].item<int>();
    const torch::Tensor input_shape = torch::slice(input_des, 0, 2, 4, 1).to(torch::kInt32);
    const int weight_bits = weight_des[0].item<int>();
    const int weight_sign = weight_des[1].item<int>();
    const torch::Tensor weight_shape = torch::slice(weight_des, 0, 2, 4, 1).to(torch::kInt32);
    TORCH_CHECK(input_shape[1].item<int>() == weight_shape[1].item<int>(), "Input and weight do not match");

    //Parse bias
    torch::Tensor _bias;
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        _bias = bias.value().to(torch::kFloat32);
        TORCH_CHECK(weight_shape[0].item<int>() == _bias.size(0), "Weight and bias do not match");
    } else {
        _bias = torch::zeros({weight_shape[0].item<int>()}, weight.options()).to(torch::kFloat32);
    }  

    // Reshape the scales and zeros
    const int batch_size = input_shape[0].item<int>();
    const int input_size = input_shape[1].item<int>();
    const int output_size = weight_shape[0].item<int>();

    torch::Tensor _input_scale = input_scale;
    if (input_scale.dim() == 0)
        _input_scale = input_scale.expand({batch_size}).contiguous();

    torch::Tensor _input_zero = input_zero;
    if (input_zero.dim() == 0)
        _input_zero = input_zero.expand({batch_size}).contiguous();

    torch::Tensor _weight_scale = weight_scale;
    if (weight_scale.dim() == 0)
        _weight_scale = weight_scale.expand({output_size}).contiguous();

    torch::Tensor _weight_zero = weight_zero;
    if (weight_zero.dim() == 0)
        _weight_zero = weight_zero.expand({output_size}).contiguous();

    return quantlinear_cuda(
        input, input_bits, input_sign, _input_scale, _input_zero,
        weight, weight_bits, weight_sign, _weight_scale, _weight_zero,
        _bias, batch_size, input_size, output_size
    );
}
