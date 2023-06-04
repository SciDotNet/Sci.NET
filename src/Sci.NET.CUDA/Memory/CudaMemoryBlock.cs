// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.CUDA.Native;
using Sci.NET.CUDA.Runtime;

namespace Sci.NET.CUDA.Memory;

/// <summary>
/// Represents an allocated block of memory on a CUDA device.
/// </summary>
/// <typeparam name="T">The type of element stored in the block.</typeparam>
[PublicAPI]
public sealed class CudaMemoryBlock<T> : IMemoryBlock<T>
    where T : unmanaged
{
    private readonly ICudaRuntimeApi _cudaRuntimeApi;
    private readonly unsafe T* _pointer;

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="count">The number of elements to allocate.</param>
    public unsafe CudaMemoryBlock(long count)
    {
        _cudaRuntimeApi = RuntimeApiResolver.Resolve();
        _pointer = _cudaRuntimeApi.Allocate<T>(count);
    }

    /// <inheritdoc />
    public long Length { get; }

    /// <inheritdoc />
    public bool IsDisposed { get; private set; }

    /// <inheritdoc />
    public unsafe IMemoryBlock<T> Copy()
    {
        var copy = new CudaMemoryBlock<T>(Length);
        _cudaRuntimeApi.CopyDeviceToDevice(ToPointer(), copy.ToPointer(), Length);
        return copy;
    }

    /// <inheritdoc />
    public unsafe SystemMemoryBlock<T> ToSystemMemory()
    {
        var copy = new SystemMemoryBlock<T>(Length);
        _cudaRuntimeApi.CopyDeviceToHost(ToPointer(), copy.ToPointer(), Length);
        return copy;
    }

    /// <inheritdoc />
    public unsafe void CopyTo(IMemoryBlock<T> destination)
    {
        if (destination is not CudaMemoryBlock<T> cudaDestination)
        {
            throw new InvalidOperationException(
                $"Destination must be a {typeof(CudaMemoryBlock<T>).Name}, but was {destination.GetType().Name}");
        }

        _cudaRuntimeApi.CopyDeviceToDevice(ToPointer(), cudaDestination.ToPointer(), Length);
    }

    /// <inheritdoc />
    public unsafe T* ToPointer()
    {
        if (!IsDisposed)
        {
            return _pointer;
        }

        throw new ObjectDisposedException(nameof(CudaMemoryBlock<T>));
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
        throw new PlatformNotSupportedException("CUDA does not support filling memory blocks.");
    }

    /// <inheritdoc />
    public unsafe void CopyFromSystemMemory(SystemMemoryBlock<T> source)
    {
        if (source.Length != Length)
        {
            throw new InvalidOperationException($"Source length must be {Length}, but was {source.Length}");
        }

        _cudaRuntimeApi.CopyHostToDevice(source.ToPointer(), ToPointer(), Length);
    }

    /// <inheritdoc />
    public void CopyFrom(T[] array)
    {
        using var systemMemoryBlock = new SystemMemoryBlock<T>(array);
        CopyFromSystemMemory(systemMemoryBlock);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="isDisposing">A value indicating if the instance is disposing.</param>
    private unsafe void Dispose(bool isDisposing)
    {
        if (!IsDisposed || !isDisposing)
        {
            IsDisposed = true;
        }

        _cudaRuntimeApi.Free(ToPointer());
        IsDisposed = true;
    }
}