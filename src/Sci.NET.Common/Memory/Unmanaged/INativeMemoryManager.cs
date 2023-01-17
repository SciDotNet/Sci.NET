// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Common.Memory.Unmanaged;

/// <summary>
/// An interface providing methods for managing native memory.
/// </summary>
[PublicAPI]
public interface INativeMemoryManager
{
    /// <summary>
    /// Allocates a block of memory of the specified size.
    /// </summary>
    /// <param name="count">The number of elements to allocate.</param>
    /// <typeparam name="T">The type of element being stored.</typeparam>
    /// <returns>A typed handle to the memory.</returns>
    public TypedMemoryHandle<T> Allocate<T>(SizeT count)
        where T : unmanaged;

    /// <summary>
    /// Frees a block of memory.
    /// </summary>
    /// <param name="handle">The handle to free.</param>
    /// <typeparam name="T">The type of element being stored.</typeparam>
    public void Free<T>(TypedMemoryHandle<T> handle)
        where T : unmanaged;

    /// <summary>
    /// Copies a block of memory from one location to another.
    /// </summary>
    /// <param name="source">The source to copy from.</param>
    /// <param name="destination">The destination to copy to.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <typeparam name="T">The type of data to copy.</typeparam>
    public void Copy<T>(TypedMemoryHandle<T> source, TypedMemoryHandle<T> destination, SizeT count)
        where T : unmanaged;

    /// <summary>
    /// Copies the contents of the array to the memory.
    /// </summary>
    /// <param name="array">The array to copy.</param>
    /// <typeparam name="T">The type of element to copy.</typeparam>
    /// <returns>A handle to the new array.</returns>
    public TypedMemoryHandle<T> CopyFromArray<T>(T[] array)
        where T : unmanaged;

    /// <summary>
    /// Copies the contents of the memory to the array to the host memory.
    /// </summary>
    /// <param name="tensorHandle">The handle to the memory.</param>
    /// <param name="tensorElementCount">The number of elements to copy.</param>
    /// <typeparam name="TNumber">The element type of the memory.</typeparam>
    /// <returns>A typed handle to the memory.</returns>
    public TypedMemoryHandle<TNumber> CopyToHostMemory<TNumber>(TypedMemoryHandle<TNumber> tensorHandle, SizeT tensorElementCount)
        where TNumber : unmanaged, INumber<TNumber>;
}