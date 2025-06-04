// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class GELUBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -4, -2, -1, -0.75f, 0, 1, 2, 60 });

        value.To(device);

        // Act
        var result = value.GELUBackward();

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { -0.032137435f, -0.14650725f, -0.085086465f, 0.0026758164f, 0.5f, 1.0850865f, 1.1465073f, 1f }, 1e-6f);
    }
}