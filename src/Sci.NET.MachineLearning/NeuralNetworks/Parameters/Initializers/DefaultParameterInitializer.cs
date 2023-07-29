// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Parameters.Initializers;

internal class DefaultParameterInitializer : IParameterInitializer
{
    public ITensor<TNumber> Initialize<TNumber>(Shape shape, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return device
            .GetTensorBackend()
            .Random
            .Uniform(shape, TNumber.CreateChecked(0.001f), TNumber.CreateChecked(0.01f));
    }
}