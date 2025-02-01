// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Power;

public class PowShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalar(IDevice device)
    {
        // Pow(2, 2) = 4
        PowScalarTest<float>(2, 2, device).Should().Be(4);
        PowScalarTest<double>(2, 2, device).Should().Be(4);
    }

    private static TNumber PowScalarTest<TNumber>(TNumber value, TNumber exponent, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IPowerFunctions<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var scalar = new Scalar<TNumber>(value);
        var exponentScalar = new Scalar<TNumber>(exponent);

        scalar.To(device);
        exponentScalar.To(device);

        return scalar.Pow(exponentScalar).Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVector(IDevice device)
    {
        // Pow([1, 2, 3], 2) = [1, 4, 9]
        PowVectorTest<float>(new float[] { 1, 2, 3 }, 2, device).Should().BeEquivalentTo(new float[] { 1, 4, 9 });
        PowVectorTest<double>(new double[] { 1, 2, 3 }, 2, device).Should().BeEquivalentTo(new double[] { 1, 4, 9 });
    }

    private static Array PowVectorTest<TNumber>(TNumber[] values, TNumber exponent, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IPowerFunctions<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values).ToVector();
        var exponentScalar = new Scalar<TNumber>(exponent);

        tensor.To(device);
        exponentScalar.To(device);

        return tensor.Pow(exponentScalar).ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrix(IDevice device)
    {
        // Pow([[1, 2, 3], [4, 5, 6]], 2) = [[1, 4, 9], [16, 25, 36]]
        PowMatrixTest<float>(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }, 2, device)
            .Should()
            .BeEquivalentTo(new float[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        PowMatrixTest<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, 2, device)
            .Should()
            .BeEquivalentTo(new double[,] { { 1, 4, 9 }, { 16, 25, 36 } });
    }

    private static Array PowMatrixTest<TNumber>(TNumber[,] values, TNumber exponent, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IPowerFunctions<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values).ToMatrix();
        var exponentScalar = new Scalar<TNumber>(exponent);

        tensor.To(device);
        exponentScalar.To(device);

        return tensor.Pow(exponentScalar).ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensor(IDevice device)
    {
        // Pow([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], 2) = [[[1, 4, 9], [16, 25, 36]], [[49, 64, 81], [100, 121, 144]]]
        PowTensorTest<float>(new float[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, 2, device)
            .Should()
            .BeEquivalentTo(new float[,,] { { { 1, 4, 9 }, { 16, 25, 36 } }, { { 49, 64, 81 }, { 100, 121, 144 } } });
        PowTensorTest<double>(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, 2, device)
            .Should()
            .BeEquivalentTo(new double[,,] { { { 1, 4, 9 }, { 16, 25, 36 } }, { { 49, 64, 81 }, { 100, 121, 144 } } });
    }

    private static Array PowTensorTest<TNumber>(TNumber[,,] values, TNumber exponent, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IPowerFunctions<TNumber>, IFloatingPointIeee754<TNumber>, ILogarithmicFunctions<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(values);
        var exponentScalar = new Scalar<TNumber>(exponent);

        tensor.To(device);
        exponentScalar.To(device);

        return tensor.Pow(exponentScalar).ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResultAndGradient_GivenPower2(IDevice device)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 2, 4 }, { 6, 8 }, { 10, 12 } };

        tensor.To(device);

        // Act
        using var result = tensor.Pow(2);

        tensor.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsMatrix().Should().BeTrue();
        result.Should().HaveEquivalentElements(new float[,] { { 1, 4 }, { 9, 16 }, { 25, 36 } });

        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResultAndGradient_GivenPower3(IDevice device)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var expectedGrad = new float[,] { { 3, 12 }, { 27, 48 }, { 75, 108 } };

        tensor.To(device);

        // Act
        using var result = tensor.Pow(3);

        tensor.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.IsMatrix().Should().BeTrue();
        result.Should().HaveEquivalentElements(new float[,] { { 1, 8 }, { 27, 64 }, { 125, 216 } });

        tensor.Gradient?.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(expectedGrad);
    }
}