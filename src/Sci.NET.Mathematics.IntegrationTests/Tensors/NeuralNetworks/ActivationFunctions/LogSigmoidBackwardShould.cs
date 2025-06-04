// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class LogSigmoidBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -4, -2, -1, 0, 1, 2, 60 });

        value.To(device);

        // Act
        var result = value.LogSigmoidBackward();

        // Assert
        result
            .Should()
            .HaveShape(7)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { 0.982013762f, 0.880797029f, 0.731058598f, 0.5f, 0.268941432f, 0.119202919f, 0 }, 1e-6f);
    }
}