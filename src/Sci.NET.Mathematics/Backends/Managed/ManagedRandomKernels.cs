// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Random;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedRandomKernels : IRandomKernels
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, long? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA5394, CA2000
        seed ??= DateTime.UtcNow.Ticks;
        MersenneTwister.Instance.Seed((uint)seed.GetHashCode());

        switch (min)
        {
            case Half:
                var resultHalfMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultHalfMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextFloat(float.CreateChecked(min), float.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultHalfMemoryBlock, shape, ManagedTensorBackend.Instance);
            case float:
                var resultFloatMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultFloatMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextFloat(float.CreateChecked(min), float.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultFloatMemoryBlock, shape, ManagedTensorBackend.Instance);
            case double:
                var resultDoubleMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultDoubleMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextDouble(double.CreateChecked(min), double.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultDoubleMemoryBlock, shape, ManagedTensorBackend.Instance);
            case byte:
                var resultByteMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultByteMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextInt(byte.CreateChecked(min), byte.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultByteMemoryBlock, shape, ManagedTensorBackend.Instance);
            case sbyte:
                var resultSByteMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultSByteMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextInt(sbyte.CreateChecked(min), sbyte.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultSByteMemoryBlock, shape, ManagedTensorBackend.Instance);
            case short:
                var resultShortMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultShortMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextInt(short.CreateChecked(min), short.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultShortMemoryBlock, shape, ManagedTensorBackend.Instance);
            case ushort:
                var resultUShortMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultUShortMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextInt(ushort.CreateChecked(min), ushort.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultUShortMemoryBlock, shape, ManagedTensorBackend.Instance);
            case int:
                var resultIntMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    resultIntMemoryBlock[i] = TNumber.CreateChecked(
                        MersenneTwister.Instance.NextInt(int.CreateChecked(min), int.CreateChecked(max)));
                }

                return new Tensor<TNumber>(resultIntMemoryBlock, shape, ManagedTensorBackend.Instance);
            default:
                throw new InvalidOperationException("Unsupported for a random number type.");
#pragma warning restore CA5394, CA2000
        }
    }
}