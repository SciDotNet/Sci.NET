// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.CUDA.Native.Extensions;

namespace Sci.NET.CUDA.Native;

internal static class RandomNativeApi
{
    public static unsafe void UniformFp32<TNumber>(IMemoryBlock<TNumber> dst, float min, float max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomFp32(
                (float*)dst.ToPointer(),
                min,
                max,
                count * sizeof(float),
                seed)
            .Guard();
    }

    public static unsafe void UniformFp64<TNumber>(IMemoryBlock<TNumber> dst, double min, double max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomFp64(
                (double*)dst.ToPointer(),
                min,
                max,
                count * sizeof(double),
                seed)
            .Guard();
    }

    public static unsafe void UniformUInt8<TNumber>(IMemoryBlock<TNumber> dst, byte min, byte max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomUInt8(
                (byte*)dst.ToPointer(),
                min,
                max,
                count * sizeof(byte),
                seed)
            .Guard();
    }

    public static unsafe void UniformUInt16<TNumber>(IMemoryBlock<TNumber> dst, ushort min, ushort max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomUInt16(
                (ushort*)dst.ToPointer(),
                min,
                max,
                count * sizeof(ushort),
                seed)
            .Guard();
    }

    public static unsafe void UniformUInt32<TNumber>(IMemoryBlock<TNumber> dst, uint min, uint max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomUInt32(
                (uint*)dst.ToPointer(),
                min,
                max,
                count * sizeof(uint),
                seed)
            .Guard();
    }

    public static unsafe void UniformUInt64<TNumber>(IMemoryBlock<TNumber> dst, ulong min, ulong max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomUInt64(
                (ulong*)dst.ToPointer(),
                min,
                max,
                count * sizeof(ulong),
                seed)
            .Guard();
    }

    public static unsafe void UniformInt8<TNumber>(IMemoryBlock<TNumber> dst, sbyte min, sbyte max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomInt8(
                (sbyte*)dst.ToPointer(),
                min,
                max,
                count * sizeof(sbyte),
                seed)
            .Guard();
    }

    public static unsafe void UniformInt16<TNumber>(IMemoryBlock<TNumber> dst, short min, short max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomInt16(
                (short*)dst.ToPointer(),
                min,
                max,
                count * sizeof(short),
                seed)
            .Guard();
    }

    public static unsafe void UniformInt32<TNumber>(IMemoryBlock<TNumber> dst, int min, int max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomInt32(
                (int*)dst.ToPointer(),
                min,
                max,
                count * sizeof(int),
                seed)
            .Guard();
    }

    public static unsafe void UniformInt64<TNumber>(IMemoryBlock<TNumber> dst, long min, long max, long count, long seed)
        where TNumber : unmanaged
    {
        RandomNativeMethods
            .UniformRandomInt64(
                (long*)dst.ToPointer(),
                min,
                max,
                count * sizeof(long),
                seed)
            .Guard();
    }
}