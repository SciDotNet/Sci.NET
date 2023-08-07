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
}