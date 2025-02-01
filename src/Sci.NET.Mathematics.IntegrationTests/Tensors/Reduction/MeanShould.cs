// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Reduction;

public class MeanShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MeanAllElements_GivenFloatMatrixAndNoAxis(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 5 / 3f, 5 / 3f }, { 5 / 3f, 5 / 3f }, { 5 / 3f, 5 / 3f } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Mean();

        result.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result.ToScalar().Value.Should().Be(3.5f);
        result.Gradient?.Should().NotBeNull();
        result.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MeanAllElements_GivenFloatMatrixAndAxis0(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 1 / 3f, 0 }, { 1 / 3f, 0 }, { 1 / 3f, 0 } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Mean([0]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 3, 4 });
    }
}