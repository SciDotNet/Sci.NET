// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Neural network extension methods for <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class NeuralNetworkExtensions
{
    /// <summary>
    /// Performs a tensor contraction on the specified tensors.
    /// </summary>
    /// <param name="input">The input <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="kernel">The kernel weights.</param>
    /// <param name="strides">The stride.</param>
    /// <param name="padding">The padding.</param>
    /// <param name="dilation">The dilation.</param>
    /// <typeparam name="TNumber">The number type for the operation.</typeparam>
    /// <returns>The convolved <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Conv2DForward<TNumber>(
        this Tensor<TNumber> input,
        Tensor<TNumber> kernel,
        int strides,
        int padding,
        int dilation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetConvolutionService()
            .Conv2D(input, kernel, strides, padding, dilation);
    }

    /// <summary>
    /// Performs a tensor contraction on the specified tensors.
    /// </summary>
    /// <param name="input">The input <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="kernel">The kernel weights.</param>
    /// <param name="strideX">The stride in the x dimension.</param>
    /// <param name="strideY">The stride in the y dimension.</param>
    /// <param name="paddingX">The padding in the x dimension.</param>
    /// <param name="paddingY">The padding in the y dimension.</param>
    /// <param name="dilationX">The dilation in the x dimension.</param>
    /// <param name="dilationY">TThe dilation in the y dimension.</param>
    /// <typeparam name="TNumber">The number type for the operation.</typeparam>
    /// <returns>The convolved <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Conv2DForward<TNumber>(
        this Tensor<TNumber> input,
        Tensor<TNumber> kernel,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetConvolutionService()
            .Conv2D(input, kernel, strideX, strideY, paddingX, paddingY, dilationX, dilationY);
    }
}