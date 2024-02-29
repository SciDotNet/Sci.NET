// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.Normalisation;

public class BatchNormShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void BatchNorm(IDevice device)
    {
        // Arrange
        var input = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }).ToMatrix();
        var scale = Tensor.FromArray<float>(new float[] { 1, 2 }).ToVector();
        var bias = Tensor.FromArray<float>(new float[] { 3, 4 }).ToVector();

        input.To(device);
        scale.To(device);
        bias.To(device);

        // Act
        var result = input.BatchNorm1dForward(scale, bias);

        // Assert
        result
            .Should()
            .HaveShape(3, 2)
            .And
            .HaveApproximatelyEquivalentElements(new float[,] { { 1.7752552032470703f, 1.5505104064941406f }, { 3.0f, 4.0f }, { 4.22474479675293f, 6.449489593505859f } }, 1e-6f);
    }
}