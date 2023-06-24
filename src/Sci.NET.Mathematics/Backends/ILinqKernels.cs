// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for LINQ kernels.
/// </summary>
[PublicAPI]
public interface ILinqKernels
{
    /// <summary>
    /// Replaces all elements of the <see cref="ITensor{TNumber}"/> with the specified value.
    /// </summary>
    /// <param name="tensor">The the <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <param name="result">The place to store the result of thee operation.</param>
    /// <param name="action">The action to perform.</param>
    /// <typeparam name="TTensor">The concrete type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Map<TTensor, TNumber>(
        ITensor<TNumber> tensor,
        ITensor<TNumber> result,
        Func<TNumber, TNumber> action)
        where TTensor : class, ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>;
}