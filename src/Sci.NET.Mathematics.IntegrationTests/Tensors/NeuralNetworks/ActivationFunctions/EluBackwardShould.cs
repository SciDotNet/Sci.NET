// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class EluBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.EluBackward(1.0f);

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { 0.1353353f, 0.3678794f, 1, 1, 1, 1, 1, 1 }, 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenDouble(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<double>(new double[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.EluBackward(1.0);

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new double[] { 0.1353352832366127, 0.3678794411714423, 1, 1, 1, 1, 1, 1 }, 1e-6);
    }
}