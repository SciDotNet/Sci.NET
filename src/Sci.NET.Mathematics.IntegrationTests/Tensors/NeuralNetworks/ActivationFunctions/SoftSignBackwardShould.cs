// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SoftSignBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -4, -2, -1, -0.75f, 0, 1, 2, 60 });

        value.To(device);

        // Act
        var result = value.SoftSignBackward();

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { 0.04f, 0.11111111f, 0.25f, 0.3265306f, 1f, 0.25f, 0.11111111f, 0.00026874497f }, 1e-6f);
    }
}