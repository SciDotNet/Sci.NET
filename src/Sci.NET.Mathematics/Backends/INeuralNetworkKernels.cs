// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for neural network kernels.
/// </summary>
[PublicAPI]
public interface INeuralNetworkKernels
{
    /// <summary>
    /// Applies a convolution to <paramref name="input"/> using <paramref name="kernels"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="input">The input image.</param>
    /// <param name="kernels">The kernels.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <param name="strideX">The stride of the operation in the x dimension.</param>
    /// <param name="strideY">The stride of the operation in the y dimension.</param>
    /// <param name="paddingX">The padding of the operation in the x dimension.</param>
    /// <param name="paddingY">The padding of the operation in the y dimension.</param>
    /// <param name="dilationX">The dilation of the operation in the x dimension.</param>
    /// <param name="dilationY">The dilation of the operation in the y dimension.</param>
    /// <typeparam name="TNumber">The number type of the operation.</typeparam>
    public void Conv2dForward<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        Tensor<TNumber> result,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the gradients of the <paramref name="input"/> and <paramref name="kernels"/> <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="input">The original input.</param>
    /// <param name="kernels">The kernel weights.</param>
    /// <param name="dOutput">The gradient of the output image.</param>
    /// <param name="dInput">The gradient of the input.</param>
    /// <param name="dKernel">The gradient of the kernel.</param>
    /// <param name="strideX">Gets the stride in the x dimension.</param>
    /// <param name="strideY">Gets the stride in the y dimension.</param>
    /// <param name="paddingX">Gets the padding in the x dimension.</param>
    /// <param name="paddingY">Gets the padding in the y dimension.</param>
    /// <param name="dilationX">Gets the dilation in the x dimension.</param>
    /// <param name="dilationY">Gets the dilation in the y dimension.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Conv2dBackward<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        Tensor<TNumber> dOutput,
        Tensor<TNumber> dInput,
        Tensor<TNumber> dKernel,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a forward pass of a 1D batch normalization operation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="scale">The scale scalar.</param>
    /// <param name="bias">The bias scalar.</param>
    /// <param name="runningMean">The running mean scalar.</param>
    /// <param name="runningVariance">The running variance scalar.</param>
    /// <param name="result">The result tensor.</param>
    /// <param name="epsilon">The epsilon value.</param>
    /// <typeparam name="TNumber">The number type of the operation.</typeparam>
    public void BatchNorm1dForward<TNumber>(
        Matrix<TNumber> input,
        Tensors.Vector<TNumber> scale,
        Tensors.Vector<TNumber> bias,
        Tensors.Vector<TNumber> runningMean,
        Tensors.Vector<TNumber> runningVariance,
        Matrix<TNumber> result,
        Scalar<TNumber> epsilon)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>;
}