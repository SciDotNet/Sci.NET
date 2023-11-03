// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Runtime;

namespace Sci.NET.CUDA.Native;

internal static class RandomNativeMethods
{
    static RandomNativeMethods()
    {
        _ = RuntimeDllImportResolver.LoadLibrary(NativeMethods.NativeLibrary, typeof(NativeMethods).Assembly);
    }

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomFp32(float* dst, float min, float max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomFp64(double* dst, double min, double max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomUInt8(byte* dst, byte min, byte max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomUInt16(ushort* dst, ushort min, ushort max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomUInt32(uint* dst, uint min, uint max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomUInt64(ulong* dst, ulong min, ulong max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomInt8(sbyte* dst, sbyte min, sbyte max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomInt16(short* dst, short min, short max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomInt32(int* dst, int min, int max, long count, long seed);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "random_uniform_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode UniformRandomInt64(long* dst, long min, long max, long count, long seed);
}