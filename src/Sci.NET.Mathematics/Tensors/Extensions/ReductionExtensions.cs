// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Provides extension methods for <see cref="ITensor{TNumber}"/> reduction operations.
/// </summary>
[PublicAPI]
public static class ReductionExtensions
{
    /// <summary>
    /// Computes the sum of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of all of the elements in the <see cref="ITensor{TNumber}"/>.</returns>
    public static ITensor<TNumber> Sum<TNumber>(this ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .Sum(tensor, axes, keepDims);
    }
}