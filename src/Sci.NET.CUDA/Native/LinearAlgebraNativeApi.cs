// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.CUDA.Native.Extensions;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.Native;

internal static class LinearAlgebraNativeApi
{
    public static unsafe void MatrixMultiplyU8<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyU8(
                (byte*)left.Memory.ToPointer(),
                (byte*)right.Memory.ToPointer(),
                (byte*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyU16<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyU16(
                (ushort*)left.Memory.ToPointer(),
                (ushort*)right.Memory.ToPointer(),
                (ushort*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyU32<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyU32(
                (uint*)left.Memory.ToPointer(),
                (uint*)right.Memory.ToPointer(),
                (uint*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyU64<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyU64(
                (ulong*)left.Memory.ToPointer(),
                (ulong*)right.Memory.ToPointer(),
                (ulong*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyI8<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyI8(
                (sbyte*)left.Memory.ToPointer(),
                (sbyte*)right.Memory.ToPointer(),
                (sbyte*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyI16<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyI16(
                (short*)left.Memory.ToPointer(),
                (short*)right.Memory.ToPointer(),
                (short*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyI32<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyI32(
                (int*)left.Memory.ToPointer(),
                (int*)right.Memory.ToPointer(),
                (int*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyI64<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyI64(
                (long*)left.Memory.ToPointer(),
                (long*)right.Memory.ToPointer(),
                (long*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyBf16<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyBf16(
                (BFloat16*)left.Memory.ToPointer(),
                (BFloat16*)right.Memory.ToPointer(),
                (BFloat16*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyFp32<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyFp32(
                (float*)left.Memory.ToPointer(),
                (float*)right.Memory.ToPointer(),
                (float*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void MatrixMultiplyFp64<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .MatrixMultiplyFp64(
                (double*)left.Memory.ToPointer(),
                (double*)right.Memory.ToPointer(),
                (double*)result.Memory.ToPointer(),
                left.Rows,
                left.Columns,
                right.Columns)
            .Guard();
    }

    public static unsafe void InnerProductFp32<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductFp32(
                (float*)left.Memory.ToPointer(),
                (float*)right.Memory.ToPointer(),
                (float*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductFp64<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductFp64(
                (double*)left.Memory.ToPointer(),
                (double*)right.Memory.ToPointer(),
                (double*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductU8<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductU8(
                (byte*)left.Memory.ToPointer(),
                (byte*)right.Memory.ToPointer(),
                (byte*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductU16<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductU16(
                (ushort*)left.Memory.ToPointer(),
                (ushort*)right.Memory.ToPointer(),
                (ushort*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductU32<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductU32(
                (uint*)left.Memory.ToPointer(),
                (uint*)right.Memory.ToPointer(),
                (uint*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductU64<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductU64(
                (ulong*)left.Memory.ToPointer(),
                (ulong*)right.Memory.ToPointer(),
                (ulong*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductI8<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductI8(
                (sbyte*)left.Memory.ToPointer(),
                (sbyte*)right.Memory.ToPointer(),
                (sbyte*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductI16<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductI16(
                (short*)left.Memory.ToPointer(),
                (short*)right.Memory.ToPointer(),
                (short*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductI32<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductI32(
                (int*)left.Memory.ToPointer(),
                (int*)right.Memory.ToPointer(),
                (int*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }

    public static unsafe void InnerProductI64<TNumber>(
        Mathematics.Tensors.Vector<TNumber> left,
        Mathematics.Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LinearAlgebraNativeMethods
            .InnerProductI64(
                (long*)left.Memory.ToPointer(),
                (long*)right.Memory.ToPointer(),
                (long*)result.Memory.ToPointer(),
                left.Length)
            .Guard();
    }
}