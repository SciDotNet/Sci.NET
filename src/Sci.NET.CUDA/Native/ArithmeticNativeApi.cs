// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.CUDA.Native.Extensions;

namespace Sci.NET.CUDA.Native;

internal static class ArithmeticNativeApi
{
    public static unsafe void AddTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddTensorBroadcastTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddTensorBroadcastTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void AddBroadcastTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AddBroadcastTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractTensorBroadcastTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractTensorBroadcastTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SubtractBroadcastTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SubtractBroadcastTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyTensorBroadcastTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .MultiplyTensorBroadcastTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void MultiplyBroadcastTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .MultiplyBroadcastTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .DivideTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .DivideTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideTensorBroadcastTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideTensorBroadcastTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorFp32(
                (float*)left.ToPointer(),
                (float*)right.ToPointer(),
                (float*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorFp64(
                (double*)left.ToPointer(),
                (double*)right.ToPointer(),
                (double*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorU8(
                (byte*)left.ToPointer(),
                (byte*)right.ToPointer(),
                (byte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)right.ToPointer(),
                (ushort*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorU32(
                (uint*)left.ToPointer(),
                (uint*)right.ToPointer(),
                (uint*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)right.ToPointer(),
                (ulong*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)right.ToPointer(),
                (sbyte*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorI16(
                (short*)left.ToPointer(),
                (short*)right.ToPointer(),
                (short*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorI32(
                (int*)left.ToPointer(),
                (int*)right.ToPointer(),
                (int*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void DivideBroadcastTensorTensorI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .DivideBroadcastTensorTensorI64(
                (long*)left.ToPointer(),
                (long*)right.ToPointer(),
                (long*)result.ToPointer(),
                m,
                n)
            .Guard();
    }

    public static unsafe void SqrtFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SqrtTensorFp32(
                (float*)left.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .SqrtTensorFp64(
                (double*)left.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtU8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorU8(
                (byte*)left.ToPointer(),
                (byte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtU16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorU16(
                (ushort*)left.ToPointer(),
                (ushort*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtU32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorU32(
                (uint*)left.ToPointer(),
                (uint*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtU64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorU64(
                (ulong*)left.ToPointer(),
                (ulong*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorI16(
                (short*)left.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorI32(
                (int*)left.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void SqrtI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .SqrtTensorI64(
                (long*)left.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void NegateFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .NegateTensorFp32(
                (float*)left.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void NegateFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .NegateTensorFp64(
                (double*)left.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void NegateI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .NegateTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void NegateI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .NegateTensorI16(
                (short*)left.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void NegateI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .NegateTensorI32(
                (int*)left.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void NegateI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .NegateTensorI64(
                (long*)left.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AbsFp32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AbsTensorFp32(
                (float*)left.ToPointer(),
                (float*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AbsFp64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ArithmeticNativeMethods
            .AbsTensorFp64(
                (double*)left.ToPointer(),
                (double*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AbsI8<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .AbsTensorI8(
                (sbyte*)left.ToPointer(),
                (sbyte*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AbsI16<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .AbsTensorI16(
                (short*)left.ToPointer(),
                (short*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AbsI32<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .AbsTensorI32(
                (int*)left.ToPointer(),
                (int*)result.ToPointer(),
                n)
            .Guard();
    }

    public static unsafe void AbsI64<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged
    {
        ArithmeticNativeMethods
            .AbsTensorI64(
                (long*)left.ToPointer(),
                (long*)result.ToPointer(),
                n)
            .Guard();
    }
}