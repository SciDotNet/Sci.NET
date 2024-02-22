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