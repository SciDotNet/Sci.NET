// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Reduction;

public class SumShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenFloatMatrixAndNoAxis(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, requiresGradient: true);
        var expectedGrad = new float[,] { { 1, 1 }, { 1, 1 }, { 1, 1 } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum();

        tensor.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result.ToScalar().Value.Should().Be(21);
        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenFloatMatrixAndAxis0(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 1, 0 }, { 1, 0 }, { 1, 0 } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum([0]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 9, 12 });
        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenFloatMatrixAndAxis1(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 1, 1 }, { 0, 0 }, { 0, 0 } };
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum([1]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 3, 7, 11 });
        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenFloatMatrixAndAxis0And1(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(Enumerable.Range(0, 60).Select(x => (float)x).ToArray()).Reshape(3, 4, 5);
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum([0, 1]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 330, 342, 354, 366, 378 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenFloatMatrixAndAxis0And2(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(Enumerable.Range(0, 60).Select(x => (float)x).ToArray()).Reshape(3, 4, 5);
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum([0, 2]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result.ToVector().ToArray().Should().BeEquivalentTo(new float[] { 330, 405, 480, 555 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenLargeIntMatrix(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<int>(Enumerable.Range(0, 60000).ToArray()).Reshape(30, 40, 50);
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsScalar().Should().BeTrue();
        result
            .ToScalar()
            .Value
            .Should()
            .Be(1799970000);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void SumAllElements_GivenLargeFloatMatrixAndAxes0And1(IDevice computeDevice)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(Enumerable.Range(0, 60000).Select(x => (float)x).ToArray()).Reshape(30, 40, 50);
        tensor.To(computeDevice);

        // Act
        using var result = tensor.Sum([0, 1]);

        result.To<CpuComputeDevice>();

        // Assert
        result.IsVector().Should().BeTrue();
        result
            .ToVector()
            .ToArray()
            .Should()
            .BeEquivalentTo(
                new float[]
                {
                    35969960,
                    35971200,
                    35972360,
                    35973600,
                    35974760,
                    35976000,
                    35977164,
                    35978400,
                    35979560,
                    35980800,
                    35981960,
                    35983200,
                    35984360,
                    35985600,
                    35986764,
                    35988000,
                    35989160,
                    35990400,
                    35991560,
                    35992800,
                    35993960,
                    35995200,
                    35996364,
                    35997600,
                    35998760,
                    36000000,
                    36001160,
                    36002400,
                    36003560,
                    36004800,
                    36005964,
                    36007200,
                    36008360,
                    36009600,
                    36010760,
                    36012000,
                    36013160,
                    36014400,
                    36015564,
                    36016800,
                    36017960,
                    36019200,
                    36020360,
                    36021600,
                    36022760,
                    36024000,
                    36025164,
                    36026400,
                    36027560,
                    36028800
                });
    }
}