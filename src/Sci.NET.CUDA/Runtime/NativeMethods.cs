// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.CUDA.Runtime.Structs;

namespace Sci.NET.CUDA.Runtime;

internal static partial class NativeMethods
{
    private const string CudaRuntimeLibName = "cudart64_12";

    [LibraryImport(CudaRuntimeLibName, EntryPoint = "cudaMalloc", SetLastError = true)]
    public static unsafe partial CudaStatusCode CudaMalloc(void** devPtr, long size);

    [LibraryImport(CudaRuntimeLibName, EntryPoint = "cudaFree", SetLastError = true)]
    public static unsafe partial CudaStatusCode CudaFree(void* devPtr);

    [LibraryImport(CudaRuntimeLibName, EntryPoint = "cudaMemcpy", SetLastError = true)]
    public static unsafe partial CudaStatusCode CudaMemcpy(void* dst, void* src, long count, CudaMemcpyKind kind);
}