// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.CUDA.Native;
using Sci.NET.Mathematics.Backends;

namespace Sci.NET.CUDA.Tensors.Backend;

internal class CudaArithmeticKernels : IArithmeticKernels
{
    public void AddTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.AddTensorTensorFp32(
                    left,
                    right,
                    result,
                    n);
                break;
            case double:
                ArithmeticNativeApi.AddTensorTensorFp64(
                    left,
                    right,
                    result,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.AddTensorTensorU8(
                    left,
                    right,
                    result,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.AddTensorTensorU16(
                    left,
                    right,
                    result,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.AddTensorTensorU32(
                    left,
                    right,
                    result,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.AddTensorTensorU64(
                    left,
                    right,
                    result,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.AddTensorTensorI8(
                    left,
                    right,
                    result,
                    n);
                break;
            case short:
                ArithmeticNativeApi.AddTensorTensorI16(
                    left,
                    right,
                    result,
                    n);
                break;
            case int:
                ArithmeticNativeApi.AddTensorTensorI32(
                    left,
                    right,
                    result,
                    n);
                break;
            case long:
                ArithmeticNativeApi.AddTensorTensorI64(
                    left,
                    right,
                    result,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void AddTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.AddTensorBroadcastTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.AddTensorBroadcastTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.AddTensorBroadcastTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.AddTensorBroadcastTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.AddTensorBroadcastTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.AddTensorBroadcastTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.AddTensorBroadcastTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.AddTensorBroadcastTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.AddTensorBroadcastTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.AddTensorBroadcastTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void AddBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.AddBroadcastTensorTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.AddBroadcastTensorTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.AddBroadcastTensorTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.AddBroadcastTensorTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.AddBroadcastTensorTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.AddBroadcastTensorTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.AddBroadcastTensorTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.AddBroadcastTensorTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.AddBroadcastTensorTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.AddBroadcastTensorTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void SubtractTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.SubtractTensorTensorFp32(
                    left,
                    right,
                    result,
                    n);
                break;
            case double:
                ArithmeticNativeApi.SubtractTensorTensorFp64(
                    left,
                    right,
                    result,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.SubtractTensorTensorU8(
                    left,
                    right,
                    result,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.SubtractTensorTensorU16(
                    left,
                    right,
                    result,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.SubtractTensorTensorU32(
                    left,
                    right,
                    result,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.SubtractTensorTensorU64(
                    left,
                    right,
                    result,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.SubtractTensorTensorI8(
                    left,
                    right,
                    result,
                    n);
                break;
            case short:
                ArithmeticNativeApi.SubtractTensorTensorI16(
                    left,
                    right,
                    result,
                    n);
                break;
            case int:
                ArithmeticNativeApi.SubtractTensorTensorI32(
                    left,
                    right,
                    result,
                    n);
                break;
            case long:
                ArithmeticNativeApi.SubtractTensorTensorI64(
                    left,
                    right,
                    result,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void SubtractTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.SubtractTensorBroadcastTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void SubtractBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.SubtractBroadcastTensorTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void MultiplyTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.MultiplyTensorTensorFp32(
                    left,
                    right,
                    result,
                    n);
                break;
            case double:
                ArithmeticNativeApi.MultiplyTensorTensorFp64(
                    left,
                    right,
                    result,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.MultiplyTensorTensorU8(
                    left,
                    right,
                    result,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.MultiplyTensorTensorU16(
                    left,
                    right,
                    result,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.MultiplyTensorTensorU32(
                    left,
                    right,
                    result,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.MultiplyTensorTensorU64(
                    left,
                    right,
                    result,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.MultiplyTensorTensorI8(
                    left,
                    right,
                    result,
                    n);
                break;
            case short:
                ArithmeticNativeApi.MultiplyTensorTensorI16(
                    left,
                    right,
                    result,
                    n);
                break;
            case int:
                ArithmeticNativeApi.MultiplyTensorTensorI32(
                    left,
                    right,
                    result,
                    n);
                break;
            case long:
                ArithmeticNativeApi.MultiplyTensorTensorI64(
                    left,
                    right,
                    result,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void MultiplyTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.MultiplyTensorBroadcastTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void MultiplyBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.MultiplyBroadcastTensorTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void DivideTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.DivideTensorTensorFp32(
                    left,
                    right,
                    result,
                    n);
                break;
            case double:
                ArithmeticNativeApi.DivideTensorTensorFp64(
                    left,
                    right,
                    result,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.DivideTensorTensorU8(
                    left,
                    right,
                    result,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.DivideTensorTensorU16(
                    left,
                    right,
                    result,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.DivideTensorTensorU32(
                    left,
                    right,
                    result,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.DivideTensorTensorU64(
                    left,
                    right,
                    result,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.DivideTensorTensorI8(
                    left,
                    right,
                    result,
                    n);
                break;
            case short:
                ArithmeticNativeApi.DivideTensorTensorI16(
                    left,
                    right,
                    result,
                    n);
                break;
            case int:
                ArithmeticNativeApi.DivideTensorTensorI32(
                    left,
                    right,
                    result,
                    n);
                break;
            case long:
                ArithmeticNativeApi.DivideTensorTensorI64(
                    left,
                    right,
                    result,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void DivideTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.DivideTensorBroadcastTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.DivideTensorBroadcastTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.DivideTensorBroadcastTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.DivideTensorBroadcastTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.DivideTensorBroadcastTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.DivideTensorBroadcastTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.DivideTensorBroadcastTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.DivideTensorBroadcastTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.DivideTensorBroadcastTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.DivideTensorBroadcastTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void DivideBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.DivideBroadcastTensorTensorFp32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case double:
                ArithmeticNativeApi.DivideBroadcastTensorTensorFp64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case byte:
                ArithmeticNativeApi.DivideBroadcastTensorTensorU8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ushort:
                ArithmeticNativeApi.DivideBroadcastTensorTensorU16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case uint:
                ArithmeticNativeApi.DivideBroadcastTensorTensorU32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case ulong:
                ArithmeticNativeApi.DivideBroadcastTensorTensorU64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case sbyte:
                ArithmeticNativeApi.DivideBroadcastTensorTensorI8(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case short:
                ArithmeticNativeApi.DivideBroadcastTensorTensorI16(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case int:
                ArithmeticNativeApi.DivideBroadcastTensorTensorI32(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            case long:
                ArithmeticNativeApi.DivideBroadcastTensorTensorI64(
                    left,
                    right,
                    result,
                    m,
                    n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void Negate<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.NegateFp32(tensor, result, n);
                break;
            case double:
                ArithmeticNativeApi.NegateFp64(tensor, result, n);
                break;
            case byte:
            case ushort:
            case uint:
            case ulong:
                tensor.CopyTo(result);
                break;
            case sbyte:
                ArithmeticNativeApi.NegateI8(tensor, result, n);
                break;
            case short:
                ArithmeticNativeApi.NegateI16(tensor, result, n);
                break;
            case int:
                ArithmeticNativeApi.NegateI32(tensor, result, n);
                break;
            case long:
                ArithmeticNativeApi.NegateI64(tensor, result, n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void Abs<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.AbsFp32(tensor, result, n);
                break;
            case double:
                ArithmeticNativeApi.AbsFp64(tensor, result, n);
                break;
            case byte:
            case ushort:
            case uint:
            case ulong:
                tensor.CopyTo(result);
                break;
            case sbyte:
                ArithmeticNativeApi.AbsI8(tensor, result, n);
                break;
            case short:
                ArithmeticNativeApi.AbsI16(tensor, result, n);
                break;
            case int:
                ArithmeticNativeApi.AbsI32(tensor, result, n);
                break;
            case long:
                ArithmeticNativeApi.AbsI64(tensor, result, n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void Sqrt<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                ArithmeticNativeApi.SqrtFp32(tensor, result, n);
                break;
            case double:
                ArithmeticNativeApi.SqrtFp64(tensor, result, n);
                break;
            case byte:
                ArithmeticNativeApi.SqrtU8(tensor, result, n);
                break;
            case ushort:
                ArithmeticNativeApi.SqrtU16(tensor, result, n);
                break;
            case uint:
                ArithmeticNativeApi.SqrtU32(tensor, result, n);
                break;
            case ulong:
                ArithmeticNativeApi.SqrtU64(tensor, result, n);
                break;
            case sbyte:
                ArithmeticNativeApi.SqrtI8(tensor, result, n);
                break;
            case short:
                ArithmeticNativeApi.SqrtI16(tensor, result, n);
                break;
            case int:
                ArithmeticNativeApi.SqrtI32(tensor, result, n);
                break;
            case long:
                ArithmeticNativeApi.SqrtI64(tensor, result, n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }
}