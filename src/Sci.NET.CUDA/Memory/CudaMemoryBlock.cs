// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Memory;
using Sci.NET.CUDA.RuntimeApi;
using Sci.NET.CUDA.RuntimeApi.Bindings.Types;

namespace Sci.NET.CUDA.Memory;

/// <summary>
/// An implementation of <see cref="IMemoryBlock{T}"/> for CUDA memory.
/// </summary>
/// <typeparam name="T">The type of memory being allocated.</typeparam>
[PublicAPI]
public sealed class CudaMemoryBlock<T> : IMemoryBlock<T>, IEquatable<CudaMemoryBlock<T>>
    where T : unmanaged
{
    private readonly unsafe T* _pointer;

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="array">The array to copy to CUDA device memory.</param>
    public unsafe CudaMemoryBlock(T[] array)
    {
        _pointer = CudaMemoryApi.CudaMalloc<T>(array.LongLength);
        Length = array.LongLength;

        using var memoryBlock = new SystemMemoryBlock<T>(array);

        CudaMemoryApi.CudaMemcpy(
            _pointer,
            memoryBlock.ToPointer(),
            array.LongLength,
            CudaMemcpyKind.HostToDevice);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="count">The number of elements to allocate.</param>
    public unsafe CudaMemoryBlock(long count)
    {
        _pointer = CudaMemoryApi.CudaMalloc<T>(count);
        Length = count;
    }

    private unsafe CudaMemoryBlock(T* pointer, long start, long length)
    {
        _pointer = (T*)Unsafe.AsPointer(ref Unsafe.Add(ref Unsafe.AsRef<T>(pointer), (nuint)start));
        Length = length;
    }

    /// <inheritdoc />
    public long Length { get; }

    /// <inheritdoc />
    public bool IsDisposed { get; }

    /// <inheritdoc />
    public ref T this[long index] =>
        throw new PlatformNotSupportedException($"{nameof(CudaMemoryBlock<T>)} does not support indexing.");

    /// <summary>
    /// Determines if the left and right <see cref="SystemMemoryBlock{T}"/>s are equal.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the two operands are equal.</returns>
    public static bool operator ==(CudaMemoryBlock<T> left, CudaMemoryBlock<T> right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Determines if the left and right <see cref="SystemMemoryBlock{T}"/>s are not equal.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the two operands are equal.</returns>
    public static bool operator !=(CudaMemoryBlock<T> left, CudaMemoryBlock<T> right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc />
    public unsafe void Dispose()
    {
        CudaMemoryApi.CudaFree(_pointer);
    }

    /// <inheritdoc />
    public unsafe IMemoryBlock<T> Slice(long start, long length)
    {
        return new CudaMemoryBlock<T>(_pointer, start, length);
    }

    /// <inheritdoc />
    public IMemoryBlock<T> Copy()
    {
        var dst = new CudaMemoryBlock<T>(Length);
        CopyTo(dst);
        return dst;
    }

    /// <inheritdoc />
    public unsafe SystemMemoryBlock<T> ToSystemMemory()
    {
        var systemMemoryBlock = new SystemMemoryBlock<T>(Length);

        CudaMemoryApi.CudaMemcpy(
            systemMemoryBlock.ToPointer(),
            _pointer,
            Length,
            CudaMemcpyKind.DeviceToHost);

        return systemMemoryBlock;
    }

    /// <inheritdoc />
    public unsafe void CopyTo(IMemoryBlock<T> destination)
    {
        if (destination is not CudaMemoryBlock<T> cudaMemoryBlock)
        {
            throw new ArgumentException("Destination must be a CudaMemoryBlock.", nameof(destination));
        }

        CudaMemoryApi.CudaMemcpy(
            cudaMemoryBlock._pointer,
            _pointer,
            Length,
            CudaMemcpyKind.DeviceToDevice);
    }

    /// <inheritdoc />
    public unsafe T* ToPointer()
    {
        return _pointer;
    }

    /// <inheritdoc />
    public unsafe T[] ToArray()
    {
        var array = new T[Length];

        CudaMemoryApi.CudaMemcpy(
            (T*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array)),
            _pointer,
            Length,
            CudaMemcpyKind.DeviceToHost);

        return array;
    }

    /// <inheritdoc />
    public unsafe void CopyFrom(T[] array)
    {
        using var memoryBlock = new SystemMemoryBlock<T>(array);

        CudaMemoryApi.CudaMemcpy(
            _pointer,
            memoryBlock.ToPointer(),
            array.LongLength,
            CudaMemcpyKind.HostToDevice);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public unsafe bool Equals(CudaMemoryBlock<T>? other)
    {
        return !IsDisposed && other is not null && _pointer == other._pointer && Length == other.Length;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is CudaMemoryBlock<T> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override unsafe int GetHashCode()
    {
        return HashCode.Combine(unchecked((int)(long)_pointer), Length);
    }
}