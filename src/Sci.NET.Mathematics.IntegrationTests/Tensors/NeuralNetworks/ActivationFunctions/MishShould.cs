// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class MishShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -6, -1.192f, 0, 1, 2, 3, 4 });
        value.To(device);

        // Act
        var result = value.Mish();

        // Assert
        result
            .Should()
            .HaveShape(7)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { -0.014853849f, -0.3088434f, 0, 0.86509836f, 1.9439591F, 2.986535f, 3.9974127f }, 1e-6f);
    }
}