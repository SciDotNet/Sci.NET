// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SoftmaxShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample1(IDevice device)
    {
        // Softmax([1, 2, 3]) = [0.090030573170380462f, 0.244728471054797646f, 0.665240955774821878f]
        var floatResult = SoftmaxTest<float>(new float[] { 1, 2, 3 }, device) as float[];
        var doubleResult = SoftmaxTest<double>(new double[] { 1, 2, 3 }, device) as double[];

        floatResult![0].Should().BeApproximately(0.09003057317038045f, 1e-6f);
        floatResult[1].Should().BeApproximately(0.24472847105479764f, 1e-6f);
        floatResult[2].Should().BeApproximately(0.665240955774821878f, 1e-6f);

        doubleResult![0].Should().BeApproximately(0.09003057317038045, 1e-6);
        doubleResult[1].Should().BeApproximately(0.24472847105479764, 1e-6);
        doubleResult[2].Should().BeApproximately(0.665240955774821878, 1e-6);
    }

    private static Array SoftmaxTest<TNumber>(TNumber[] values, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values);

        tensor.To(device);

        return tensor.Softmax().ToArray();
    }
}