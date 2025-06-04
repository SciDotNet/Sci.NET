// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class MishBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -7, -3, -1.419f, 0, 1, 2, 3, 4 });
        value.To(device);

        // Act
        var result = value.MishBackward();

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { -0.0054659364F, -0.09339317F, -0.050504386F, 0.6f, 1.0490361F, 1.069318F, 1.0211062F, 1.0044336F }, 1e-6f);
    }
}