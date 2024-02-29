// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SigmoidShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensor(IDevice device)
    {
        // Sig(0.5) = 0.622459331201854564638
        ScalarSigmoidTest<float>(0.5f, device).Should().BeApproximately(0.622459331201854564638f, 1e-6f);
        ScalarSigmoidTest<double>(0.5, device).Should().BeApproximately(0.622459331201854564638, 1e-6);

        // Sig(0.0) = 0.5
        ScalarSigmoidTest<float>(0.0f, device).Should().BeApproximately(0.5f, 1e-6f);
        ScalarSigmoidTest<double>(0.0, device).Should().BeApproximately(0.5, 1e-6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargerTensor(IDevice device)
    {
        const float expected = 0.622459331201854564638f;

        // Arrange
        var tensor = Tensor.FillWith(0.5f, new Shape(8, 8, 8));

        tensor.To(device);

        // Act
        var result = tensor.Sigmoid();

        // Assert
        result
            .Should()
            .HaveShape(8, 8, 8)
            .And
            .HaveAllElementsApproximately(expected, 1e-6f);
    }

    private static TNumber ScalarSigmoidTest<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.Sigmoid().ToScalar().Value;
    }
}