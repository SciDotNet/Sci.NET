// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Numerics;
using Sci.NET.Common.Runtime;

namespace Sci.NET.CUDA.Native;

internal static class LinearAlgebraNativeMethods
{
    static LinearAlgebraNativeMethods()
    {
        _ = RuntimeDllImportResolver.LoadLibrary(NativeMethods.NativeLibrary, typeof(NativeMethods).Assembly, "CUDA");
    }

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyU8(
        byte* left,
        byte* right,
        byte* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyU16(
        ushort* left,
        ushort* right,
        ushort* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyU32(
        uint* left,
        uint* right,
        uint* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyU64(
        ulong* left,
        ulong* right,
        ulong* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyI16(
        short* left,
        short* right,
        short* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyI32(
        int* left,
        int* right,
        int* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyI64(
        long* left,
        long* right,
        long* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_bf16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyBf16(
        BFloat16* left,
        BFloat16* right,
        BFloat16* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyFp32(
        float* left,
        float* right,
        float* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "matrix_multiply_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MatrixMultiplyFp64(
        double* left,
        double* right,
        double* result,
        int leftRows,
        int leftColumns,
        int rightColumns);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductFp32(float* a, float* b, float* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductFp64(double* a, double* b, double* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductU8(byte* a, byte* b, byte* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductU16(ushort* a, ushort* b, ushort* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductU32(uint* a, uint* b, uint* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductU64(ulong* a, ulong* b, ulong* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductI8(sbyte* a, sbyte* b, sbyte* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductI16(short* a, short* b, short* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductI32(int* a, int* b, int* c, int n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "inner_product_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode InnerProductI64(long* a, long* b, long* c, int n);
}