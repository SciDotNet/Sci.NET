// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Native.Extensions;
using Sci.NET.CUDA.RuntimeApi.Types;

namespace Sci.NET.CUDA.Native;

internal static class NativeApi
{
    public static int CudaGetDeviceCount()
    {
        var count = 0;
        NativeMethods.GetCudaDeviceCount(ref count).Guard();

        return count;
    }

    public static CudaDeviceProperties GetDeviceProperties(int deviceIdx)
    {
        var props = default(CudaDeviceProps);
        NativeMethods.GetCudaDeviceProps(deviceIdx, ref props).Guard();

        return CudaDeviceProperties.FromNativeProps(props, deviceIdx);
    }

    public static unsafe T* Allocate<T>(long count)
        where T : unmanaged
    {
        var ptr = default(void*);
        NativeMethods.AllocateMemory(ref ptr, new IntPtr(count * sizeof(T))).Guard();

        return (T*)ptr;
    }

    public static unsafe void Free<T>(T* ptr)
        where T : unmanaged
    {
        NativeMethods.FreeMemory(ptr).Guard();
    }

    public static unsafe void CopyDeviceToDevice<T>(T* dst, T* src, long count)
        where T : unmanaged
    {
        NativeMethods.CopyMemoryDeviceToDevice(dst, src, count * sizeof(T)).Guard();
    }

    public static unsafe void CopyDeviceToHost<T>(T* dst, T* src, long count)
        where T : unmanaged
    {
        NativeMethods.CopyMemoryToHost(dst, src, count * sizeof(T)).Guard();
    }

    public static unsafe void CopyHostToDevice<T>(T* dst, T* src, long count)
        where T : unmanaged
    {
        NativeMethods.CopyMemoryToDevice(dst, src, count * sizeof(T)).Guard();
    }
}