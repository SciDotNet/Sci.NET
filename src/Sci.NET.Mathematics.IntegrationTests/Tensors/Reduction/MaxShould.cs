// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Linq;
using Sci.NET.Common.Random;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Reduction;

public class MaxShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MaxAllElements_GivenFloatMatrixAndNoAxis(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Max();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result.ToScalar().Value.Should().Be(6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MaxAllElements_GivenFloatMatrixAndAxis0(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Max([0]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 5, 6 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MaxAllElements_GivenLargeRandomArray(IDevice computeDevice)
    {
        // Arrange
        var shape = new int[] { 50, 60, 70 };
        var array = Enumerable.Range(1, shape.Product()).ToArray();
        array.Shuffle();
        using var tensor = Tensor.FromArray<int>(array).Reshape(shape);
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Max();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result.ToScalar().Value.Should().Be(shape.Product());
    }
}