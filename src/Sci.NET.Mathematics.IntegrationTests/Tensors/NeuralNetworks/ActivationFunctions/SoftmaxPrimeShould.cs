// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SoftmaxPrimeShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample1(IDevice device)
    {
        // Softmax'([1, 2, 3]) = [0.090030573170380462f, 0.244728471054797646f, 0.665240955774821878f]
        SoftmaxPrimeTest<float>(new float[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new float[] { 0.08192507F, 0.18483646F, 0.22269543F });
        SoftmaxPrimeTest<double>(new double[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new double[] { 0.08192506906499324, 0.18483644650997874, 0.22269542653462335 });
    }

    private static Array SoftmaxPrimeTest<TNumber>(TNumber[] values, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values);

        tensor.To(device);

        return tensor.SoftmaxPrime().ToArray();
    }
}