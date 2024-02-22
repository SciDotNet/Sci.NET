// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class HardSigmoidShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -2, -1, -0.5f, 0, 0.5f, 1, 2 });

        value.To(device);

        // Act
        var result = value.HardSigmoid();

        // Assert
        result
            .Should()
            .HaveShape(7)
            .And
            .HaveEquivalentElements(new float[] { 0, 0, -0.25f, 0, 0.25f, 1, 1 });
    }
}