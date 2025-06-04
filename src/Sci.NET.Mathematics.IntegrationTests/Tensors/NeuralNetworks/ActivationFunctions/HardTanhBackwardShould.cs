// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class HardTanhBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -2, -1, 0, 1, 2, 3, 4, 5 });

        value.To(device);

        // Act
        var result = value.HardTanhBackward(-2, 2);

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveEquivalentElements(new float[] { 0, 1, 1, 1, 0, 0, 0, 0 });
    }
}