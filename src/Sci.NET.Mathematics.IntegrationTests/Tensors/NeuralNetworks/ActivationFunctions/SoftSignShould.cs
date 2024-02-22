// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SoftSignShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -4, -2, -1, 0, 1, 2, 60 });

        value.To(device);

        // Act
        var result = value.SoftSign();

        // Assert
        result
            .Should()
            .HaveShape(7)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { -0.8f, -0.6666667f, -0.5f, 0, 0.5f, 0.6666667f, 0.9836066f }, 1e-6f);
    }
}