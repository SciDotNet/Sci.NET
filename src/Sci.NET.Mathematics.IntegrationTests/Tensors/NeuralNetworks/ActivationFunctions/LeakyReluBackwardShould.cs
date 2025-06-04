// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class LeakyReluBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.LeakyReLUBackward(0.5f);

        // Assert
        result.ToArray().Should().BeEquivalentTo(new float[] { 0.5f, 0.5f, 0.5f, 1, 1, 1, 1, 1 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenDouble(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<double>(new double[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.LeakyReLUBackward(0.5f);

        // Assert
        result.ToArray().Should().BeEquivalentTo(new double[] { 0.5, 0.5, 0.5, 1, 1, 1, 1, 1 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenInt(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<int>(new int[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.LeakyReLUBackward(2);

        // Assert
        result.ToArray().Should().BeEquivalentTo(new int[] { 2, 2, 2, 1, 1, 1, 1, 1 });
    }
}