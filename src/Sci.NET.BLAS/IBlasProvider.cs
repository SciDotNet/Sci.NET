// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;

namespace Sci.NET.BLAS;

/// <summary>
/// An interface for basic linear algebra subprograms.
/// </summary>
[PublicAPI]
public interface IBlasProvider
{
    /// <summary>
    /// Allocate a memory block of the specified size.
    /// </summary>
    /// <param name="count">The number of elements to allocate.</param>
    /// <typeparam name="T">The type of memory to allocate.</typeparam>
    /// <returns>A typed handle to the allocation.</returns>
    public TypedMemoryHandle<T> Allocate<T>(long count)
        where T : unmanaged, INumber<T>;
}