// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for permutation kernels.
/// </summary>
[PublicAPI]
public interface IPermutationKernels
{
    /// <summary>
    /// Permutes the specified tensor according to the specified permutation.
    /// </summary>
    /// <param name="source">The tensor to permute.</param>
    /// <param name="result">The result of the permutation.</param>
    /// <param name="permutation">The permutation to apply.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Permute<TNumber>(ITensor<TNumber> source, ITensor<TNumber> result, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>;
}