// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Linq;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Reduction;

public class MinShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MinAllElements_GivenFloatMatrixAndNoAxis(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, requiresGradient: true);
        var expectedGrad = new float[,] { { 1, 0 }, { 0, 0 }, { 0, 0 } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Min();

        result.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result.ToScalar().Value.Should().Be(1);
        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void MinAllElements_GivenFloatMatrixAndAxis0(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 1, 1 }, { 0, 0 }, { 0, 0 } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Min([0]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 1, 2 });
        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnMinValue_GivenLargeIntMatrix(IDevice computeDevice)
    {
        // Arrange
        var shape = new int[] { 100, 100, 50 };
        using var tensor = Tensor.FromArray<int>(Enumerable.Range(1, shape.Product()).ToArray()).Reshape(shape).ToTensor();
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Min();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result.ToScalar().Value.Should().Be(1);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnMinValues_GivenLargeIntTensorAndAxis1(IDevice computeDevice)
    {
        // Arrange
        var shape = new int[] { 100, 100, 50 };
        using var tensor = Tensor.FromArray<int>(Enumerable.Range(1, shape.Product()).ToArray()).Reshape(shape).ToTensor();
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Min([1]);

        result.To<CpuComputeDevice>();

        var expectedTensor = Tensor.FromArray<int>(Enumerable.Range(1, 5000).ToArray()).Reshape(100, 50).ToTensor();

        // Assert
        result.IsMatrix().Should().BeTrue();
        result
            .ToMatrix()
            .Should()
            .HaveShape(100, 50)
            .And
            .HaveEquivalentElements(expectedTensor.ToArray());
    }
}