// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Memory;

/// <summary>
/// Represents a contiguous region of memory.
/// </summary>
/// <typeparam name="T">The type of memory stored within that region of memory.</typeparam>
[PublicAPI]
public interface IMemoryBlock<T> : IDisposable
    where T : unmanaged
{
    /// <summary>
    /// Gets the length of the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    public long Length { get; }

    /// <summary>
    /// Gets a value indicating whether the <see cref="IMemoryBlock{T}"/> has been disposed.
    /// </summary>
    public bool IsDisposed { get; }

    /// <summary>
    /// Gets the element at the specified index.
    /// </summary>
    /// <param name="index">The index of the element to get.</param>
    public ref T this[long index] { get; }

    /// <summary>
    /// Gets a <see cref="IMemoryBlock{T}"/> that represents a slice of the current <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="start">The start index.</param>
    /// <param name="length">The length of the slice.</param>
    /// <returns>A slice of the current <see cref="IMemoryBlock{T}"/>.</returns>
    public IMemoryBlock<T> Slice(long start, long length);

    /// <summary>
    /// Copies the contents of the <see cref="IMemoryBlock{T}"/> to a new <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <returns>A copy of the current <see cref="IMemoryBlock{T}"/> instance.</returns>
    public IMemoryBlock<T> Copy();

    /// <summary>
    /// Copies the <see cref="IMemoryBlock{T}"/> to a <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IMemoryBlock{T}"/> as a <see cref="SystemMemoryBlock{T}"/>.</returns>
    public SystemMemoryBlock<T> ToSystemMemory();

    /// <summary>
    /// Copies the contents of the <see cref="IMemoryBlock{T}"/> to the specified <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="destination">The <see cref="IMemoryBlock{T}"/> to copy to.</param>
    public void CopyTo(IMemoryBlock<T> destination);

    /// <summary>
    /// Converts the <see cref="IMemoryBlock{T}"/> to a pointer.
    /// </summary>
    /// <returns>A pointer to the <see cref="IMemoryBlock{T}"/>.</returns>
    unsafe T* ToPointer();

    /// <summary>
    /// Converts the <see cref="IMemoryBlock{T}"/> to an array.
    /// </summary>
    /// <returns>The current instance converted to an array.</returns>
    public T[] ToArray();

    /// <summary>
    /// Copies the contents of the specified array to the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="array">The array to copy from.</param>
    public void CopyFrom(T[] array);
}