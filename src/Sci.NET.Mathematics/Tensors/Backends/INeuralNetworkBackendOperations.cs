// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends;

/// <summary>
/// An interface for neural network backend operations.
/// </summary>
[PublicAPI]
public interface INeuralNetworkBackendOperations
{
    /// <summary>
    /// Applies a 2D convolutional operation to the <paramref name="input"/> <see cref="ITensor{TNumber}"/>
    /// using the <paramref name="kernel"/> <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="kernel">The kernel <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="strideX">The stride of the operation in the x dimension.</param>
    /// <param name="strideY">The stride of the operation in the y dimension.</param>
    /// <param name="dilationX">The dilation of the operation in the x dimension.</param>
    /// <param name="dilationY">The dilation of the operation in the y dimension.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the convolution operation on the <paramref name="input"/> <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Conv2d<TNumber>(
        ITensor<TNumber> input,
        ITensor<TNumber> kernel,
        int strideX,
        int strideY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>;
}