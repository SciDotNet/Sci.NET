// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks;

/// <summary>
/// An interface for convolution services.
/// </summary>
[PublicAPI]
public interface IConvolutionService
{
    /// <summary>
    /// Performs a 2D convolution.
    /// </summary>
    /// <param name="input">The input image.</param>
    /// <param name="kernels">The kernel weights.</param>
    /// <param name="strideX">The stride in the x dimension.</param>
    /// <param name="strideY">The stride in the y dimension.</param>
    /// <param name="paddingX">The padding in the x dimension.</param>
    /// <param name="paddingY">The padding in the y dimension.</param>
    /// <param name="dilationX">The dilation in the x dimension.</param>
    /// <param name="dilationY">The dilation in the y dimension.</param>
    /// <typeparam name="TNumber">The number type for the operation.</typeparam>
    /// <returns>The convolved input.</returns>
    public Tensor<TNumber> Conv2D<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a 2D convolution.
    /// </summary>
    /// <param name="input">The input image.</param>
    /// <param name="kernels">The kernel weights.</param>
    /// <param name="stride">The stride.</param>
    /// <param name="padding">The padding.</param>
    /// <param name="dilation">The dilation.</param>
    /// <typeparam name="TNumber">The number type for the operation.</typeparam>
    /// <returns>The convolved image.</returns>
    public Tensor<TNumber> Conv2D<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        int stride,
        int padding,
        int dilation)
        where TNumber : unmanaged, INumber<TNumber>;
}