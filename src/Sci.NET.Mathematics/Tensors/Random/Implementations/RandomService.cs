// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors.Random.Implementations;

internal class RandomService : IRandomService
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, int? seed = null, IDevice? device = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.Uniform(
                shape,
                min,
                max,
                seed);
    }

    public ITensor<TNumber> Uniform<TNumber, TDevice>(Shape shape, TNumber min, TNumber max, int? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
        where TDevice : IDevice, new()
    {
        return Uniform(shape, min, max, seed, new TDevice());
    }
}