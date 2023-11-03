// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.CUDA.Native;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.Tensors.Backend;

internal class CudaLinearAlgebraKernels : ILinearAlgebraKernels
{
    public void MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case BFloat16:
                LinearAlgebraNativeApi.MatrixMultiplyBf16(left, right, result);
                break;
            case float:
                LinearAlgebraNativeApi.MatrixMultiplyFp32(left, right, result);
                break;
            case double:
                LinearAlgebraNativeApi.MatrixMultiplyFp64(left, right, result);
                break;
            case byte:
                LinearAlgebraNativeApi.MatrixMultiplyU8(left, right, result);
                break;
            case ushort:
                LinearAlgebraNativeApi.MatrixMultiplyU16(left, right, result);
                break;
            case uint:
                LinearAlgebraNativeApi.MatrixMultiplyU32(left, right, result);
                break;
            case ulong:
                LinearAlgebraNativeApi.MatrixMultiplyU64(left, right, result);
                break;
            case sbyte:
                LinearAlgebraNativeApi.MatrixMultiplyI8(left, right, result);
                break;
            case short:
                LinearAlgebraNativeApi.MatrixMultiplyI16(left, right, result);
                break;
            case int:
                LinearAlgebraNativeApi.MatrixMultiplyI32(left, right, result);
                break;
            case long:
                LinearAlgebraNativeApi.MatrixMultiplyI64(left, right, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for matrix multiplication.");
        }
    }

    public void InnerProduct<TNumber>(Mathematics.Tensors.Vector<TNumber> left, Mathematics.Tensors.Vector<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                LinearAlgebraNativeApi.InnerProductFp32(left, right, result);
                break;
            case double:
                LinearAlgebraNativeApi.InnerProductFp64(left, right, result);
                break;
            case byte:
                LinearAlgebraNativeApi.InnerProductU8(left, right, result);
                break;
            case ushort:
                LinearAlgebraNativeApi.InnerProductU16(left, right, result);
                break;
            case uint:
                LinearAlgebraNativeApi.InnerProductU32(left, right, result);
                break;
            case ulong:
                LinearAlgebraNativeApi.InnerProductU64(left, right, result);
                break;
            case sbyte:
                LinearAlgebraNativeApi.InnerProductI8(left, right, result);
                break;
            case short:
                LinearAlgebraNativeApi.InnerProductI16(left, right, result);
                break;
            case int:
                LinearAlgebraNativeApi.InnerProductI32(left, right, result);
                break;
            case long:
                LinearAlgebraNativeApi.InnerProductI64(left, right, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for matrix multiplication.");
        }
    }
}