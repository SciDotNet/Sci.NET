// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Memory;

/// <summary>
/// Represents a contiguous region of memory.
/// </summary>
/// <typeparam name="T">The type of memory stored within that region of memory.</typeparam>
[PublicAPI]
public interface IMemoryBlock<T> : IReferenceCounted, IDisposable
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
    /// Gets or sets the element at the specified index.
    /// </summary>
    /// <param name="index">The index of the element to get or set.</param>
    public ref T this[long index] { get; }

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
    public unsafe T* ToPointer();

    /// <summary>
    /// Converts the <see cref="IMemoryBlock{T}"/> to an array.
    /// </summary>
    /// <returns>The current instance converted to an array.</returns>
    public T[] ToArray();

    /// <summary>
    /// Fills the <see cref="IMemoryBlock{T}"/> with the specified value.
    /// </summary>
    /// <param name="value">The value to fill the <see cref="IMemoryBlock{T}"/> with.</param>
    public void Fill(T value);

    /// <summary>
    /// Copies the contents of the <see cref="SystemMemoryBlock{T}"/> to this <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="source">The source memory block.</param>
    public void CopyFromSystemMemory(SystemMemoryBlock<T> source);

    /// <summary>
    /// Copies the contents of the specified array to the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="array">The array to copy from.</param>
    public void CopyFrom(T[] array);

    /// <summary>
    /// Writes the contents of the <see cref="IMemoryBlock{T}"/> to the specified stream.
    /// </summary>
    /// <param name="stream">The stream to write to.</param>
    public void WriteTo(Stream stream);

    /// <summary>
    /// Copies the contents of the specified <see cref="IMemoryBlock{T}"/> to the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="handle">The memory block to copy from.</param>
    /// <param name="srcIdx">The source index to start from.</param>
    /// <param name="dstIdx">The destination index to start from.</param>
    /// <param name="count">The number of elements to copy.</param>
    public void BlockCopyFrom(IMemoryBlock<T> handle, long srcIdx, long dstIdx, long count);

    /// <summary>
    /// Copies the contents of the specified <see cref="Span{T}"/> to the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="buffer">The buffer to copy from.</param>
    /// <param name="srcIdx">The source index to start from in bytes.</param>
    /// <param name="dstIdx">The destination index to start from in bytes.</param>
    /// <param name="count">The number of bytes to copy.</param>
    public void BlockCopyFrom(Span<byte> buffer, int srcIdx, int dstIdx, int count);

    /// <summary>
    /// Frees the memory associated with the <see cref="IMemoryBlock{T}"/> irrespective of the reference count.
    /// </summary>
    public void UnsafeFreeMemory();
}