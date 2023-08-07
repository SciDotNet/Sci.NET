// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common;
using Sci.NET.Common.Runtime;
using Sci.NET.CUDA.RuntimeApi.Types;

namespace Sci.NET.CUDA.Native;

internal static class NativeMethods
{
    public const string NativeLibrary = "Sci.NET.CUDA.Native";

    static NativeMethods()
    {
        _ = RuntimeDllImportResolver.LoadLibrary(NativeLibrary, typeof(NativeMethods).Assembly);
    }

    [DllImport(NativeLibrary, EntryPoint = "allocate_memory", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AllocateMemory(ref void* ptr, SizeT size);

    [DllImport(NativeLibrary, EntryPoint = "free_memory", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode FreeMemory(void* ptr);

    [DllImport(NativeLibrary, EntryPoint = "copy_memory_to_device", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CopyMemoryToDevice(void* dst, void* src, SizeT size);

    [DllImport(NativeLibrary, EntryPoint = "copy_memory_to_host", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CopyMemoryToHost(void* dst, void* src, SizeT size);

    [DllImport(NativeLibrary, EntryPoint = "copy_memory_device_to_device", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CopyMemoryDeviceToDevice(void* dst, void* src, SizeT size);

    [DllImport(NativeLibrary, EntryPoint = "get_cuda_device_props", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode GetCudaDeviceProps(int deviceIdx, ref CudaDeviceProps props);

    [DllImport(NativeLibrary, EntryPoint = "get_cuda_device_count", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode GetCudaDeviceCount(ref int count);

    [DllImport(NativeLibrary, EntryPoint = "set_cublas_tensor_core_mode", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SetCublasTensorCoreMode(bool enabled);
}