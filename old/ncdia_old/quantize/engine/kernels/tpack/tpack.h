#pragma once
#include <torch/extension.h>

#include <vector>


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
    bool sign);


/**
 * Unpacks the given vector of tensors into a tensor.
 *
 * @param x The vector of tensors to be unpacked.
 * @param des The description tensor of the packing.
 * @return The unpacked tensor.
 */
torch::Tensor tunpack(
    torch::Tensor x,
    torch::Tensor des);
