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
                ArithmeticNativeApi.AddTensorTensorFp32(left, right, result, n);
                break;
            case double:
                ArithmeticNativeApi.AddTensorTensorFp64(left, right, result, n);
                break;
            case byte:
                ArithmeticNativeApi.AddTensorTensorU8(left, right, result, n);
                break;
            case ushort:
                ArithmeticNativeApi.AddTensorTensorU16(left, right, result, n);
                break;
            case uint:
                ArithmeticNativeApi.AddTensorTensorU32(left, right, result, n);
                break;
            case ulong:
                ArithmeticNativeApi.AddTensorTensorU64(left, right, result, n);
                break;
            case sbyte:
                ArithmeticNativeApi.AddTensorTensorI8(left, right, result, n);
                break;
            case short:
                ArithmeticNativeApi.AddTensorTensorI16(left, right, result, n);
                break;
            case int:
                ArithmeticNativeApi.AddTensorTensorI32(left, right, result, n);
                break;
            case long:
                ArithmeticNativeApi.AddTensorTensorI64(left, right, result, n);
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
                ArithmeticNativeApi.AddTensorBroadcastTensorFp32(left, right, result, m, n);
                break;
            case double:
                ArithmeticNativeApi.AddTensorBroadcastTensorFp64(left, right, result, m, n);
                break;
            case byte:
                ArithmeticNativeApi.AddTensorBroadcastTensorU8(left, right, result, m, n);
                break;
            case ushort:
                ArithmeticNativeApi.AddTensorBroadcastTensorU16(left, right, result, m, n);
                break;
            case uint:
                ArithmeticNativeApi.AddTensorBroadcastTensorU32(left, right, result, m, n);
                break;
            case ulong:
                ArithmeticNativeApi.AddTensorBroadcastTensorU64(left, right, result, m, n);
                break;
            case sbyte:
                ArithmeticNativeApi.AddTensorBroadcastTensorI8(left, right, result, m, n);
                break;
            case short:
                ArithmeticNativeApi.AddTensorBroadcastTensorI16(left, right, result, m, n);
                break;
            case int:
                ArithmeticNativeApi.AddTensorBroadcastTensorI32(left, right, result, m, n);
                break;
            case long:
                ArithmeticNativeApi.AddTensorBroadcastTensorI64(left, right, result, m, n);
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
                ArithmeticNativeApi.AddBroadcastTensorTensorFp32(left, right, result, m, n);
                break;
            case double:
                ArithmeticNativeApi.AddBroadcastTensorTensorFp64(left, right, result, m, n);
                break;
            case byte:
                ArithmeticNativeApi.AddBroadcastTensorTensorU8(left, right, result, m, n);
                break;
            case ushort:
                ArithmeticNativeApi.AddBroadcastTensorTensorU16(left, right, result, m, n);
                break;
            case uint:
                ArithmeticNativeApi.AddBroadcastTensorTensorU32(left, right, result, m, n);
                break;
            case ulong:
                ArithmeticNativeApi.AddBroadcastTensorTensorU64(left, right, result, m, n);
                break;
            case sbyte:
                ArithmeticNativeApi.AddBroadcastTensorTensorI8(left, right, result, m, n);
                break;
            case short:
                ArithmeticNativeApi.AddBroadcastTensorTensorI16(left, right, result, m, n);
                break;
            case int:
                ArithmeticNativeApi.AddBroadcastTensorTensorI32(left, right, result, m, n);
                break;
            case long:
                ArithmeticNativeApi.AddBroadcastTensorTensorI64(left, right, result, m, n);
                break;
            default:
                throw new PlatformNotSupportedException("Unsupported type for the CUDA backend.");
        }
    }

    public void SubtractTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void SubtractTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void SubtractBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void MultiplyTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void MultiplyTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void MultiplyBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void DivideTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void DivideTensorBroadcastTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void DivideBroadcastTensorTensor<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result, long m, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void Negate<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void Abs<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    public void Sqrt<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }
}