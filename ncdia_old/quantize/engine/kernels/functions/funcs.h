#pragma once 
#include <torch/extension.h>

#include <vector>


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
    const c10::optional<torch::Tensor> & bias={},
    const int mode=0);


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
    const c10::optional<torch::Tensor> & bias);
