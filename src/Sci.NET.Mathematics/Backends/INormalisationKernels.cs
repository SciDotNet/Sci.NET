// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for normalisation kernels.
/// </summary>
[PublicAPI]
public interface INormalisationKernels
{
    /// <summary>
    /// Clips the values of the <see cref="ITensor{TNumber}"/> to the specified range.
    /// </summary>
    /// <param name="tensor">The the <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <param name="result">The place to store the result of thee operation.</param>
    /// <param name="min">The minimum value to clip to.</param>
    /// <param name="max">The maximum value to clip to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Clip<TNumber>(
        ITensor<TNumber> tensor,
        ITensor<TNumber> result,
        TNumber min,
        TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a backward pass of a 1D batch normalization operation.
    /// </summary>
    /// <param name="tensor">The tensor to operate on.</param>
    /// <param name="result">The result tensor.</param>
    /// <param name="min">The minimum value to clip to.</param>
    /// <param name="max">The maximum value to clip to.</param>
    /// <typeparam name="TNumber">The number type of the operation.</typeparam>
    public void ClipBackward<TNumber>(ITensor<TNumber> tensor, Tensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;
}