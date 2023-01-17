// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common;
using Sci.NET.CUDA.RuntimeApi.Bindings.Extensions;
using Sci.NET.CUDA.RuntimeApi.Bindings.Types;

namespace Sci.NET.CUDA.RuntimeApi.Bindings;

// Do not pass types by reference
// Names should differ by more than just case
// Use DefaultDllImportSearchPaths for P/Invoke
#pragma warning disable CA1045, CA1708, CA5392
internal static class NativeMethods
{
    private const string CudaRuntimeLibName = "cudart64_12";

    public static void CudaGetDeviceProperties(ref CudaDeviceProps props, int deviceIndex)
    {
        cudaGetDeviceProperties(ref props, deviceIndex).Guard();
    }

    public static void CudaGetDeviceCount(ref int count)
    {
        cudaGetDeviceCount(ref count).Guard();
    }

    public static void CudaSetDevice(int index)
    {
        cudaSetDevice(index).Guard();
    }

    public static void CudaMalloc(ref nuint devPtr, SizeT size)
    {
        cudaMalloc(ref devPtr, size).Guard();
    }

    public static void CudaFree(nuint devPtr)
    {
        cudaFree(devPtr).Guard();
    }

    public static void CudaMemcpy(nuint dst, nuint src, SizeT size, CudaMemcpyKind kind)
    {
        var sizet = (ulong)size.ToInt64();

        cudaMemcpy(dst, src, new nuint(sizet), kind).Guard();
    }

    [DllImport(CudaRuntimeLibName)]
    private static extern CudaStatusCode cudaGetDeviceProperties(ref CudaDeviceProps props, int deviceIndex);

    [DllImport(CudaRuntimeLibName)]
    private static extern CudaStatusCode cudaGetDeviceCount(ref int count);

    [DllImport(CudaRuntimeLibName)]
    private static extern CudaStatusCode cudaSetDevice(int index);

    [DllImport(CudaRuntimeLibName)]
    private static extern CudaStatusCode cudaMalloc(ref nuint devPtr, SizeT size);

    [DllImport(CudaRuntimeLibName)]
    private static extern CudaStatusCode cudaFree(nuint devPtr);

    [DllImport(CudaRuntimeLibName)]
    private static extern CudaStatusCode cudaMemcpy(
        nuint dstPointer,
        nuint srcPointer,
        nuint size,
        CudaMemcpyKind kind);
}
#pragma warning restore CA1045, CA1708