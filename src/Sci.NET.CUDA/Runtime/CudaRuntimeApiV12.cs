// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Native;
using Sci.NET.CUDA.Runtime.Extensions;
using Sci.NET.CUDA.Runtime.Structs;

namespace Sci.NET.CUDA.Runtime;

internal class CudaRuntimeApiV12 : ICudaRuntimeApi
{
    public unsafe T* Allocate<T>(long count)
        where T : unmanaged
    {
        var ptr = (long*)0;
        NativeMethods.CudaMalloc((void**)&ptr, count * sizeof(T)).EnsureSuccess();

        return (T*)ptr;
    }

    public unsafe void Free<T>(T* memoryPtr)
        where T : unmanaged
    {
        NativeMethods.CudaFree(memoryPtr).EnsureSuccess();
    }

    public unsafe void CopyDeviceToDevice<T>(T* source, T* destination, long count)
        where T : unmanaged
    {
        NativeMethods.CudaMemcpy(destination, source, count * sizeof(T), CudaMemcpyKind.DeviceToDevice).EnsureSuccess();
    }

    public unsafe void CopyDeviceToHost<T>(T* source, T* destination, long count)
        where T : unmanaged
    {
        NativeMethods.CudaMemcpy(destination, source, count * sizeof(T), CudaMemcpyKind.DeviceToHost).EnsureSuccess();
    }

    public unsafe void CopyHostToDevice<T>(T* source, T* destination, long count)
        where T : unmanaged
    {
        NativeMethods.CudaMemcpy(destination, source, count * sizeof(T), CudaMemcpyKind.HostToDevice).EnsureSuccess();
    }
}