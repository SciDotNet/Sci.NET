// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.InteropServices;
using Sci.NET.Common;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.Unmanaged;
using Sci.NET.CUDA.RuntimeApi;
using Sci.NET.CUDA.RuntimeApi.Bindings.Types;

namespace Sci.NET.CUDA.Memory;

/// <summary>
/// A CUDA implementation of <see cref="INativeMemoryManager"/>.
/// </summary>
[PublicAPI]
public class CudaMemoryManager : INativeMemoryManager
{
    /// <inheritdoc />
    public TypedMemoryHandle<T> Allocate<T>(SizeT count)
        where T : unmanaged
    {
        return CudaMemoryApi.CudaMalloc<T>(count);
    }

    /// <inheritdoc />
    public void Free<T>(TypedMemoryHandle<T> handle)
        where T : unmanaged
    {
        CudaMemoryApi.CudaFree(handle);
    }

    /// <inheritdoc />
    public void Copy<T>(TypedMemoryHandle<T> source, TypedMemoryHandle<T> destination, SizeT count)
        where T : unmanaged
    {
        CudaMemoryApi.CudaMemcpy(destination, source, count, CudaMemcpyKind.DeviceToDevice);
    }

    /// <inheritdoc />
    public unsafe TypedMemoryHandle<T> CopyFromArray<T>(T[] array)
        where T : unmanaged
    {
        var dst = Allocate<T>(array.Length);

#pragma warning disable RCS1176
        fixed (T* ptr = &array[0])
#pragma warning restore RCS1176
        {
            CudaMemoryApi.CudaMemcpy(
                dst,
                new TypedMemoryHandle<T>(ptr),
                array.Length,
                CudaMemcpyKind.HostToDevice);
        }

        return dst;
    }

    /// <inheritdoc />
#pragma warning disable CA1822
    public unsafe TypedMemoryHandle<TNumber> CopyToHostMemory<TNumber>(
        TypedMemoryHandle<TNumber> tensorHandle,
        SizeT tensorElementCount)
        where TNumber : unmanaged, INumber<TNumber>
#pragma warning restore CA1822
    {
        var handle =
            new TypedMemoryHandle<TNumber>(
                (TNumber*)NativeMemory.AllocZeroed(tensorElementCount.ToUIntPtr(), (nuint)sizeof(TNumber)));

        CudaMemoryApi.CudaMemcpy(
            handle,
            tensorHandle,
            tensorElementCount,
            CudaMemcpyKind.DeviceToHost);

        return handle;
    }
}