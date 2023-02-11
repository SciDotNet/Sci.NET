// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors.Manipulation;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

internal class DefaultRandomBackendOperations : IRandomBackendOperations
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, long seed)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA5394, CA2000
        var random = new System.Random(seed.GetHashCode());

        switch (min)
        {
            case float:
                var floatMemoryBlock = new SystemMemoryBlock<float>(shape.ElementCount);

                for (var i = 0; i < shape.ElementCount; i++)
                {
                    floatMemoryBlock[i] = random.NextSingle();
                }

                return new Tensor<float>(floatMemoryBlock, shape).Cast<float, TNumber>();

            case double:
                var doubleMemoryBlock = new SystemMemoryBlock<double>(shape.ElementCount);

                for (var i = 0; i < shape.ElementCount; i++)
                {
                    doubleMemoryBlock[i] = random.NextDouble();
                }

                return new Tensor<double>(doubleMemoryBlock, shape).Cast<double, TNumber>();

            case int:
                var intMemoryBlock = new SystemMemoryBlock<int>(shape.ElementCount);

                for (var i = 0; i < shape.ElementCount; i++)
                {
                    intMemoryBlock[i] = random.Next();
                }

                return new Tensor<int>(intMemoryBlock, shape).Cast<int, TNumber>();

            case long:
                var longMemoryBlock = new SystemMemoryBlock<long>(shape.ElementCount);

                for (var i = 0; i < shape.ElementCount; i++)
                {
                    longMemoryBlock[i] = random.NextInt64();
                }

                return new Tensor<long>(longMemoryBlock, shape).Cast<long, TNumber>();

            default:
                throw new InvalidOperationException("Unsupported for a random number type.");
#pragma warning restore CA5394, CA2000
        }
    }
}