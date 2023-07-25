// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedRandomKernels : IRandomKernels
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, long? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA5394, CA2000

        var range = max - min;

        switch (min)
        {
            case Half:
                var resultHalfMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.NextDouble();
                    resultHalfMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultHalfMemoryBlock, shape, ManagedTensorBackend.Instance);
            case float:
                var resultFloatMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.NextDouble();
                    resultFloatMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultFloatMemoryBlock, shape, ManagedTensorBackend.Instance);
            case double:
                var resultDoubleMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.NextDouble();
                    resultDoubleMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultDoubleMemoryBlock, shape, ManagedTensorBackend.Instance);
            case byte:
                var resultByteMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.Next();
                    resultByteMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultByteMemoryBlock, shape, ManagedTensorBackend.Instance);
            case sbyte:
                var resultSByteMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.Next();
                    resultSByteMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultSByteMemoryBlock, shape, ManagedTensorBackend.Instance);
            case short:
                var resultShortMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.Next();
                    resultShortMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultShortMemoryBlock, shape, ManagedTensorBackend.Instance);
            case ushort:
                var resultUShortMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.Next();
                    resultUShortMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultUShortMemoryBlock, shape, ManagedTensorBackend.Instance);
            case int:
                var resultIntMemoryBlock = new SystemMemoryBlock<TNumber>(shape.ElementCount);

                for (var i = 0L; i < shape.ElementCount; i++)
                {
                    var random = Random.Shared.Next();
                    resultIntMemoryBlock[i] = min + (TNumber.CreateChecked(random) * range);
                }

                return new Tensor<TNumber>(resultIntMemoryBlock, shape, ManagedTensorBackend.Instance);
            default:
                throw new InvalidOperationException("Unsupported for a random number type.");
#pragma warning restore CA5394, CA2000
        }
    }
}