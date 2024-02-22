// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class LogSigmoidShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -4, -2, -1, 0, 1, 2, 60 });

        value.To(device);

        // Act
        var result = value.LogSigmoid();

        // Assert
        result
            .Should()
            .HaveShape(7)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { -4.01814985f, -2.12692809f, -1.31326163f, -0.693147182f, -0.313261658f, -0.126928061f, 0 }, 1e-4f);
    }
}