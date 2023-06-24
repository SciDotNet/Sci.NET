// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// Represents a storage backend.
/// </summary>
[PublicAPI]
public interface ITensorStorageKernels
{
    /// <summary>
    /// Allocates a tensor handle with the given length.
    /// </summary>
    /// <param name="shape">The length of the allocation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    /// <returns>A handle to the allocated memory.</returns>
    public IMemoryBlock<TNumber> Allocate<TNumber>(Shape shape)
        where TNumber : unmanaged, INumber<TNumber>;
}