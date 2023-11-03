// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.LowLevel;
using Sci.NET.CUDA.Native;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.Tensors.Backend;

internal class CudaRandomKernels : IRandomKernels
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, int? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, new CudaTensorBackend());

        switch (TNumber.Zero)
        {
            case float:
                RandomNativeApi
                    .UniformFp32(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, float>(),
                        max.ReinterpretCast<TNumber, float>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case double:
                RandomNativeApi
                    .UniformFp64(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, double>(),
                        max.ReinterpretCast<TNumber, double>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case byte:
                RandomNativeApi
                    .UniformUInt8(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, byte>(),
                        max.ReinterpretCast<TNumber, byte>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case ushort:
                RandomNativeApi
                    .UniformUInt16(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, ushort>(),
                        max.ReinterpretCast<TNumber, ushort>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case uint:
                RandomNativeApi
                    .UniformUInt32(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, uint>(),
                        max.ReinterpretCast<TNumber, uint>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case ulong:
                RandomNativeApi
                    .UniformUInt64(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, ulong>(),
                        max.ReinterpretCast<TNumber, ulong>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case sbyte:
                RandomNativeApi
                    .UniformInt8(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, sbyte>(),
                        max.ReinterpretCast<TNumber, sbyte>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case short:
                RandomNativeApi
                    .UniformInt16(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, short>(),
                        max.ReinterpretCast<TNumber, short>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case int:
                RandomNativeApi
                    .UniformInt32(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, int>(),
                        max.ReinterpretCast<TNumber, int>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            case long:
                RandomNativeApi
                    .UniformInt64(
                        tensor.Handle,
                        min.ReinterpretCast<TNumber, long>(),
                        max.ReinterpretCast<TNumber, long>(),
                        shape.ElementCount,
                        seed ?? 0);
                break;
            default:
                throw new NotSupportedException("Unsupported type for uniform random number generation.");
        }

        return tensor;
    }
}