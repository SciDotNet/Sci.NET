// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Random;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedRandomKernels : IRandomKernels
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, int? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);

        Prng.Uniform(
            tensor.Memory,
            min,
            max,
            seed);

        return tensor;
    }
}