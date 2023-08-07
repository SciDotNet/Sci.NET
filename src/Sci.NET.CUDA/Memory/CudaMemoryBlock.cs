// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.CUDA.Native;

namespace Sci.NET.CUDA.Memory;

/// <summary>
/// A <see cref="IMemoryBlock{T}"/> implementation for CUDA.
/// </summary>
/// <typeparam name="T">The type of the <see cref="IMemoryBlock{T}"/>.</typeparam>
[PublicAPI]
public class CudaMemoryBlock<T> : IMemoryBlock<T>
    where T : unmanaged
{
    private readonly unsafe T* _reference;

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="length">The length of the <see cref="CudaMemoryBlock{T}"/>.</param>
    public unsafe CudaMemoryBlock(long length)
    {
        _reference = NativeApi.Allocate<T>(length);
        Length = length;
        IsDisposed = false;
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="CudaMemoryBlock{T}"/> class.
    /// </summary>
    ~CudaMemoryBlock()
    {
        Dispose(false);
    }

    /// <inheritdoc />
    public long Length { get; }

    /// <inheritdoc />
    public bool IsDisposed { get; private set; }

    /// <inheritdoc />
    public void Rent(Guid id)
    {
    }

    /// <inheritdoc />
    public void Release(Guid id)
    {
    }

    /// <inheritdoc />
    public unsafe IMemoryBlock<T> Copy()
    {
        var copy = new CudaMemoryBlock<T>(Length);
        NativeApi.CopyDeviceToDevice(copy._reference, _reference, Length);
        return copy;
    }

    /// <inheritdoc />
    public unsafe SystemMemoryBlock<T> ToSystemMemory()
    {
        var systemMemory = new SystemMemoryBlock<T>(Length);
        NativeApi.CopyDeviceToHost(systemMemory.ToPointer(), _reference, Length);
        return systemMemory;
    }

    /// <inheritdoc />
    public unsafe void CopyTo(IMemoryBlock<T> destination)
    {
        if (destination is not CudaMemoryBlock<T> cudaDestination)
        {
            throw new ArgumentException("Destination must be a CudaMemoryBlock<T>.", nameof(destination));
        }

        if (destination.Length != Length)
        {
            throw new ArgumentException("Source length must match destination length.", nameof(destination));
        }

        NativeApi.CopyDeviceToDevice(cudaDestination._reference, _reference, Length);
    }

    /// <inheritdoc />
    public unsafe T* ToPointer()
    {
        return _reference;
    }

    /// <inheritdoc />
    public T[] ToArray()
    {
        using var systemMemory = ToSystemMemory();
        return systemMemory.ToArray();
    }

    /// <inheritdoc />
    public void Fill(T value)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public unsafe void CopyFromSystemMemory(SystemMemoryBlock<T> source)
    {
        if (source.Length != Length)
        {
            throw new ArgumentException("Source length must match destination length.", nameof(source));
        }

        NativeApi.CopyHostToDevice(_reference, source.ToPointer(), source.Length);
    }

    /// <inheritdoc />
    public void CopyFrom(T[] array)
    {
        if (array.Length != Length)
        {
            throw new ArgumentException("Source length must match destination length.", nameof(array));
        }

        using var systemMemory = new SystemMemoryBlock<T>(array);
        CopyFromSystemMemory(systemMemory);
    }

    /// <inheritdoc />
    public void WriteTo(Stream stream)
    {
        using var systemMemory = ToSystemMemory();
        systemMemory.WriteTo(stream);
    }

    /// <inheritdoc />
    public unsafe void BlockCopyFrom(IMemoryBlock<T> handle, long srcIdx, long dstIdx, long count)
    {
        if (handle is not CudaMemoryBlock<T> cudaHandle)
        {
            throw new ArgumentException("Source must be a CudaMemoryBlock<T>.", nameof(handle));
        }

        NativeApi.CopyDeviceToDevice(_reference + dstIdx, cudaHandle._reference + srcIdx, count);
    }

    /// <inheritdoc />
    public void BlockCopyFrom(Span<byte> buffer, int srcIdx, int dstIdx, int count)
    {
        using var handle = new SystemMemoryBlock<T>(Length);
        handle.BlockCopyFrom(buffer, srcIdx, dstIdx, count);
        CopyFromSystemMemory(handle);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases the unmanaged resources used by the <see cref="CudaMemoryBlock{T}"/> and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    protected virtual unsafe void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();

        if (!IsDisposed && disposing)
        {
            IsDisposed = true;
        }
    }

    private unsafe void ReleaseUnmanagedResources()
    {
        NativeApi.Free(_reference);
    }
}