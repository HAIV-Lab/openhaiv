/**
 * @file tpack.cu
 * @brief This file contains the CUDA implementation of the packing and unpacking functions.
 * @version 0.1.0
 * @date 2024-01-05
 */

#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include "tpack.h"

#define CHECK_NBITS(b) TORCH_CHECK(b > 0 && b <= 8, #b " must be in the range (0, 8]")
#define CHECK_RANGE(xmin, xmax, min, max) TORCH_CHECK(xmin >= min && xmax <= max, "The input tensor is out of range.")
#define CHECK_LENGTH(x, min) TORCH_CHECK(x.size(0) >= min, "The description is too short, which should be at least " #min ".")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/**
 * CUDA kernel for packing the given tensor into a vector of tensors.
 *
 * @param x The input tensor to be packed.
 * @param x_out The output tensor.
 * @param n_elements The number of elements in the input tensor.
 * @param n_bits The number of bits to quantize the tensor.
 * @param offset The offset of the input tensor.
 */
template <typename scalar_t>
__global__ void tpack_cuda_kernel(
    const scalar_t* __restrict__ x,
    unsigned char* __restrict__ x_out,
    int n_elements,
    int n_bits,
    unsigned char offset)
{
    // Get the index of the starting element.
    const int start_index = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    for (int i = 0; i < 8; i++)
    {
        // Get the current index.
        const int index = start_index + i;

        // Check if the index is valid.
        if (index < n_elements)
        {
            // Get the current element and convert float to unsigned int8.
            unsigned char element = (char)x[index];
            element += offset;

            // Get the start bit index.
            const int bit_index = index * n_bits;

            // Get the current byte index.
            const int byte_index = bit_index / 8;

            // Get the current bit offset.
            const int bit_offset = bit_index % 8;

            // Get the current byte.
            unsigned char byte = x_out[byte_index];

            // Set the current bit.
            byte |= element << bit_offset;

            // Set the current byte.
            x_out[byte_index] = byte;

            if (bit_offset + n_bits > 8)
            {
                // Get the current byte.
                unsigned char byte = x_out[byte_index + 1];

                // Set the current bit.
                byte |= element >> (8 - bit_offset);

                // Set the current byte.
                x_out[byte_index + 1] = byte;
            }
        }
    }
}


/**
 * Packing the given vector into torch.uint8 with no bits wasted on CUDA.
 * 
 * @param x The input tensor to be packed, which is a flattened tensor.
 * @param x_out The output tensor with the same device as the input tensor.
 * @param n_elements The number of elements in the input tensor.
 * @param n_bits The number of bits to quantize the tensor.
 * @param sign A flag indicating whether the input tensor is signed or unsigned.
 */
void tpack_cuda(
    torch::Tensor x,
    torch::Tensor x_out,
    int n_elements,
    int n_bits,
    bool sign)
{
    // Check the input tensor.
    CHECK_CUDA(x);
    CHECK_CUDA(x_out);

    // Get the offset.
    unsigned char offset = 0;
    if (sign) { 
        offset = 1 << (n_bits - 1);
    }

    // Get the number of threads per block.
    auto threads_per_block = 1024;

    // Get the number of blocks.
    auto blocks = ((n_elements + 7 ) / 8 + threads_per_block - 1) / threads_per_block;

    // Call the CUDA kernel.
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "tpack_cuda", ([&] {
        tpack_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            x_out.data_ptr<unsigned char>(),
            n_elements,
            n_bits,
            offset);
    }));
}


/**
 * Packing the given vector into torch.uint8 with no bits wasted.
 * 
 * @param x The input tensor to be packed, which is a flattened tensor.
 * @param x_out The output tensor with the same device as the input tensor.
 * @param n_elements The number of elements in the input tensor.
 * @param n_bits The number of bits to quantize the tensor.
 * @param sign A flag indicating whether the input tensor is signed or unsigned.
 */
void tpack_cpu(
    torch::Tensor x,
    torch::Tensor x_out,
    int n_elements,
    int n_bits,
    bool sign)
{
    // Get the offset.
    unsigned char offset = 0;
    if (sign) { 
        offset = 1 << (n_bits - 1);
    }

    // Execute the packing.
    for (int i = 0; i < n_elements; i++)
    {
        // Get the current element.
        unsigned char element = (char)x[i].item<float>();
        element += offset;

        // Get the start bit index.
        auto bit_index = i * n_bits;

        // Get the current byte index.
        auto byte_index = bit_index / 8;

        // Get the current bit offset.
        auto bit_offset = bit_index % 8;

        // Get the current byte.
        auto byte = x_out[byte_index].item<unsigned char>();

        // Set the current bit.
        byte |= element << bit_offset;

        // Set the current byte.
        x_out[byte_index] = byte;

        if (bit_offset + n_bits > 8)
        {
            // Get the current byte.
            auto byte = x_out[byte_index + 1].item<unsigned char>();

            // Set the current bit.
            byte |= element >> (8 - bit_offset);

            // Set the current byte.
            x_out[byte_index + 1] = byte;
        }
    }
}


/**
 * Packs the given tensor into a vector of tensors.
 *
 * @param x The input tensor to be packed.
 * @param n_bits The number of bits to quantize the tensor.
 * @param sign A flag indicating whether the input tensor is signed or unsigned.
 * @return A vector of tensors containing the packed representation of the input tensor,
 *         and a description tensor of the packing with the following format:
 *         [n_bits, sign, shape[0], shape[1], ...]
 */
std::vector<torch::Tensor> tpack(
    torch::Tensor x,
    int n_bits,
    bool sign)
{
    // Check the input tensor.
    CHECK_NBITS(n_bits);
    CHECK_CONTIGUOUS(x);
    if (sign) {
        CHECK_RANGE(x.min().item<float>(), x.max().item<float>(), -(1 << (n_bits - 1)), (1 << (n_bits - 1)) - 1);
    } else {
        CHECK_RANGE(x.min().item<float>(), x.max().item<float>(), 0, (1 << n_bits) - 1);
    }
    
    // Get the shape of the input tensor.
    auto shape = x.sizes();

    // Get the number of elements in the input tensor.
    auto n_elements = x.numel();

    // Create the output tensor with the same device as the input tensor.
    auto n_out_elements = (n_elements * n_bits + 7) / 8;
    auto x_out = torch::zeros({n_out_elements}, torch::kByte).to(x.device());

    // Create the description tensor.
    signed long size = 2 + shape.size();
    auto des = torch::zeros({size}, torch::kInt).to(x.device());

    // Set the description tensor.
    des[0] = n_bits;

    if (sign) { des[1] = 1; } else { des[1] = 0; }

    for (int i = 0; i < shape.size(); i++) { 
        des[2 + i] = shape[i];
    }

    // Choose kernel to execute.
    if (x.device().is_cuda()) {
        // Call the CUDA kernel.
        tpack_cuda(
            x.flatten(), x_out,
            n_elements, n_bits, sign);
    } else {
        // Call the CPU kernel.
        tpack_cpu(
            x.flatten(), x_out,
            n_elements, n_bits, sign);
    }

    // Return the output tensor and the description tensor.
    return {x_out, des};
}


/**
 * CUDA kernel for unpacking the given vector into torch.int8 (uint8).
 *
 * @param x The vector of tensors to be unpacked, which is a flattened tensor.
 * @param x_out The output tensor with the same device as the input tensor.
 * @param n_elements The number of elements in the output tensor.
 * @param n_bits The number of bits to quantize the tensor.
 * @param offset The offset of the input tensor.
 */
template <typename scalar_t>
__global__ void tunpack_cuda_kernel(
    const unsigned char* __restrict__ x,
    scalar_t* __restrict__ x_out,
    int n_elements,
    int n_bits,
    unsigned char offset)
{
    // Get the index of the starting element.
    const int start_index = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    for (int i = 0; i < 8; i++)
    {
        // Get the current index.
        const int index = start_index + i;

        // Check if the index is valid.
        if (index < n_elements)
        {
            // Get the start bit index.
            const int bit_index = index * n_bits;

            // Get the current byte index.
            const int byte_index = bit_index / 8;

            // Get the current bit offset.
            const int bit_offset = bit_index % 8;

            // Get the current byte.
            unsigned char byte = x[byte_index];

            // Get the current element.
            unsigned char element = (byte >> bit_offset) & ((1 << n_bits) - 1);

            if (bit_offset + n_bits > 8)
            {
                // Get the current byte.
                unsigned char byte = x[byte_index + 1];

                // Get the current element.
                element |= (byte << (8 - bit_offset)) & ((1 << n_bits) - 1);
            }

            // Set the current element.
            element -= offset;
            x_out[index] = (scalar_t)element;
        }
    }
}


/**
 * CUDA kernel for unpacking the given vector into torch.int8 (uint8).
 *
 * @param x The vector of tensors to be unpacked, which is a flattened tensor.
 * @param x_out The output tensor with the same device as the input tensor.
 * @param n_elements The number of elements in the output tensor.
 * @param n_bits The number of bits to quantize the tensor.
 * @param sign A flag indicating whether the input tensor is signed or unsigned.
 */
void tunpack_cuda(
    torch::Tensor x,
    torch::Tensor x_out,
    int n_elements,
    int n_bits,
    bool sign)
{
    // Check the input tensor.
    CHECK_CUDA(x);
    CHECK_CUDA(x_out);

    // Get the offset.
    unsigned char offset = 0;
    if (sign) { 
        offset = 1 << (n_bits - 1);
    }

    // Get the number of threads per block.
    auto threads_per_block = 1024;

    // Get the number of blocks.
    auto blocks = ((n_elements + 7 ) / 8 + threads_per_block - 1) / threads_per_block;

    // Call the CUDA kernel.
    AT_DISPATCH_ALL_TYPES(x_out.scalar_type(), "tunpack_cuda", ([&] {
        tunpack_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
            x.data_ptr<unsigned char>(),
            x_out.data_ptr<scalar_t>(),
            n_elements,
            n_bits,
            offset);
    }));
}


/**
 * Unpacking the given vector into torch.int8 (uint8) on CPU.
 *
 * @param x The vector of tensors to be unpacked, which is a flattened tensor.
 * @param x_out The output tensor with the same device as the input tensor.
 * @param n_elements The number of elements in the output tensor.
 * @param n_bits The number of bits to quantize the tensor.
 * @param sign A flag indicating whether the input tensor is signed or unsigned.
 */
void tunpack_cpu(
    torch::Tensor x,
    torch::Tensor x_out,
    int n_elements,
    int n_bits,
    bool sign)
{
    // Get the offset.
    unsigned char offset = 0;
    if (sign) { 
        offset = 1 << (n_bits - 1);
    }

    // Execute the unpacking.
    for (int i = 0; i < n_elements; i++)
    {
        // Get the start bit index.
        auto bit_index = i * n_bits;

        // Get the current byte index.
        auto byte_index = bit_index / 8;

        // Get the current bit offset.
        auto bit_offset = bit_index % 8;

        // Get the current byte.
        auto byte = x[byte_index].item<unsigned char>();

        // Get the current element.
        unsigned char element = (byte >> bit_offset) & ((1 << n_bits) - 1);

        if (bit_offset + n_bits > 8)
        {
            // Get the current byte.
            auto byte = x[byte_index + 1].item<unsigned char>();

            // Get the current element.
            element |= (byte << (8 - bit_offset)) & ((1 << n_bits) - 1);
        }

        // Set the current element.
        element -= offset;
        if (sign) {
            x_out[i] = (signed char)element;
        } else {
            x_out[i] = (unsigned char)element;
        }
    }
}


/**
 * Unpacks the given vector of tensors into a tensor.
 *
 * @param x The vector of tensors to be unpacked.
 * @param des The description tensor of the packing.
 * @return The unpacked tensor.
 */
torch::Tensor tunpack(
    torch::Tensor x,
    torch::Tensor des)
{
    // Get the number of bits.
    CHECK_LENGTH(des, 3);
    auto n_bits = des[0].item<int>();

    // Check the input tensor.
    CHECK_NBITS(n_bits);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dtype() == torch::kByte, "The input tensor must be torch.uint8.");

    // Get the sign flag.
    auto sign = des[1].item<int>();

    // Get the shape of the output tensor.
    auto shape = torch::slice(des, 0, 2, des.numel(), 1);

    // Get the number of elements in the output tensor.
    auto n_elements = shape.prod().item<int>();

    // Create the output tensor with the same device as the input tensor.
    auto x_out = torch::zeros({n_elements}, torch::kByte).to(x.device());
    if (sign) {
        x_out = x_out.to(torch::kChar);
    }

    // Choose kernel to execute.
    if (x.device().is_cuda()) {
        // Call the CUDA kernel.
        tunpack_cuda(
            x.flatten(), x_out,
            n_elements, n_bits, sign);
    } else {
        // Call the CPU kernel.
        tunpack_cpu(
            x.flatten(), x_out,
            n_elements, n_bits, sign);
    }

    // Return the reshaped output tensor.
    std::vector<int64_t> shape_vec;
    for (int i = 0; i < shape.numel(); i++) {
        shape_vec.push_back(shape[i].item<int64_t>());
    }
    return x_out.reshape(shape_vec);
}
