// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common;
using Sci.NET.CUDA.RuntimeApi.Bindings;
using Sci.NET.CUDA.RuntimeApi.Bindings.Types;

namespace Sci.NET.CUDA.RuntimeApi;

/// <summary>
/// Bindings to the CUDA Runtime API.
/// </summary>
[PublicAPI]
public static class CudaMemoryApi
{
    /// <summary>
    /// Allocates memory on the device.
    /// </summary>
    /// <typeparam name="T">The type of element being stored.</typeparam>
    /// <param name="count">The size of the memory block to sdn_allocate.</param>
    /// <returns>A pointer to the allocated memory.</returns>
    public static unsafe T* CudaMalloc<T>(SizeT count)
        where T : unmanaged
    {
        var ptr = default(nuint);
        NativeMethods.CudaMalloc(ref ptr, count.ToInt64() * sizeof(T));
        return (T*)ptr;
    }

    /// <summary>
    /// Copies memory to and from the device.
    /// </summary>
    /// <typeparam name="T">The type of memory being copied.</typeparam>
    /// <param name="dst">The destination address.</param>
    /// <param name="src">The source address.</param>
    /// <param name="count">The length of memory to copy.</param>
    /// <param name="kind">The direction of the copy.</param>
    public static unsafe void CudaMemcpy<T>(
        T* dst,
        T* src,
        SizeT count,
        CudaMemcpyKind kind)
        where T : unmanaged
    {
        NativeMethods.CudaMemcpy((nuint)dst, (nuint)src, count.ToInt64() * sizeof(T), kind);
    }

    /// <summary>
    /// Frees memory on the device.
    /// </summary>
    /// <typeparam name="T">The type of memory being freed.</typeparam>
    /// <param name="handle">Handle to the memory to free.</param>
    public static unsafe void CudaFree<T>(T* handle)
        where T : unmanaged
    {
        NativeMethods.CudaFree((nuint)handle);
    }
}