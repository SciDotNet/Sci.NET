// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SoftPlusBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -4, -2, -1, -0.75f, 0, 1, 2, 60 });

        value.To(device);

        // Act
        var result = value.SoftPlusBackward();

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { 0.017986208f, 0.11920291f, 0.2689414f, 0.32082126f, 0.49999997f, 0.73105854f, 0.880797f, 0.99999994f }, 1e-6f);
    }
}