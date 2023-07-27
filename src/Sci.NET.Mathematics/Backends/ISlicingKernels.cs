// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for slicing kernels.
/// </summary>
[PublicAPI]
public interface ISlicingKernels
{
    /// <summary>
    /// Realizes a slice from a source <see cref="ITensor{TNumber}"/> to a destination <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="source">The source <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="destination">The destination <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    public void RealizeFromSliceShape<TNumber>(ITensor<TNumber> source, ITensor<TNumber> destination)
        where TNumber : unmanaged, INumber<TNumber>;
}