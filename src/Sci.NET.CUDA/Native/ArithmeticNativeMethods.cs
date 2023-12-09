// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Runtime;

namespace Sci.NET.CUDA.Native;

internal static class ArithmeticNativeMethods
{
    static ArithmeticNativeMethods()
    {
        _ = RuntimeDllImportResolver.LoadLibrary(NativeMethods.NativeLibrary, typeof(NativeMethods).Assembly, "CUDA");
    }

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long count);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_tensor_broadcast_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddTensorBroadcastTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "add_broadcast_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AddBroadcastTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_tensor_broadcast_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractTensorBroadcastTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "subtract_broadcast_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SubtractBroadcastTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_tensor_broadcast_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyTensorBroadcastTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "multiply_broadcast_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode MultiplyBroadcastTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_tensor_broadcast_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideTensorBroadcastTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorFp32(
        float* left,
        float* right,
        float* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorFp64(
        double* left,
        double* right,
        double* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorU8(
        byte* left,
        byte* right,
        byte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorU16(
        ushort* left,
        ushort* right,
        ushort* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorU32(
        uint* left,
        uint* right,
        uint* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorU64(
        ulong* left,
        ulong* right,
        ulong* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorI8(
        sbyte* left,
        sbyte* right,
        sbyte* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorI16(
        short* left,
        short* right,
        short* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorI32(
        int* left,
        int* right,
        int* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "divide_broadcast_tensor_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode DivideBroadcastTensorTensorI64(
        long* left,
        long* right,
        long* result,
        long m,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "negate_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode NegateTensorFp32(
        float* input,
        float* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "negate_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode NegateTensorFp64(
        double* input,
        double* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "negate_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode NegateTensorI8(
        sbyte* input,
        sbyte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "negate_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode NegateTensorI16(
        short* input,
        short* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "negate_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode NegateTensorI32(
        int* input,
        int* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "negate_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode NegateTensorI64(
        long* input,
        long* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "abs_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AbsTensorFp32(
        float* input,
        float* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "abs_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AbsTensorFp64(
        double* input,
        double* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "abs_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AbsTensorI8(
        sbyte* input,
        sbyte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "abs_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AbsTensorI16(
        short* input,
        short* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "abs_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AbsTensorI32(
        int* input,
        int* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "abs_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AbsTensorI64(
        long* input,
        long* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorFp32(
        float* input,
        float* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorFp64(
        double* input,
        double* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_u8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorU8(
        byte* input,
        byte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_u16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorU16(
        ushort* input,
        ushort* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_u32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorU32(
        uint* input,
        uint* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_u64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorU64(
        ulong* input,
        ulong* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_i8", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorI8(
        sbyte* input,
        sbyte* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_i16", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorI16(
        short* input,
        short* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_i32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorI32(
        int* input,
        int* result,
        long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sqrt_tensor_i64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SqrtTensorI64(
        long* input,
        long* result,
        long n);
}