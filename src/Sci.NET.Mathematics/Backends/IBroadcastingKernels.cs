// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for broadcasting kernels.
/// </summary>
[PublicAPI]
public interface IBroadcastingKernels
{
    /// <summary>
    /// Broadcasts a <see cref="ITensor{TNumber}"/> to a new <see cref="Shape"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to broadcast.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="strides">The strides to use for broadcasting.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void Broadcast<TNumber>(
        ITensor<TNumber> tensor,
        ITensor<TNumber> result,
        long[] strides)
        where TNumber : unmanaged, INumber<TNumber>;
}