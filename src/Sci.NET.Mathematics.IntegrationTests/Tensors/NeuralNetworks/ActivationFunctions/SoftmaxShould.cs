// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SoftmaxShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample1(IDevice device)
    {
        // Softmax([1, 2, 3]) = [0.090030573170380462f, 0.244728471054797646f, 0.665240955774821878f]
        SoftmaxTest<float>(new float[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new float[] { 0.090030573170380462f, 0.24472847105479767f, 0.665240955774821878f });
        SoftmaxTest<double>(new double[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new double[] { 0.090030573170380462, 0.24472847105479767, 0.665240955774821878 });
    }

    private static Array SoftmaxTest<TNumber>(TNumber[] values, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values);

        tensor.To(device);

        return tensor.Softmax().ToArray();
    }
}