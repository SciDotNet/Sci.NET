// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Power;

public class ExpShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalar(IDevice device)
    {
        ExpScalarTest<float>(1, device).Should().BeApproximately(2.718281828459045f, 1e-6f);
        ExpScalarTest<double>(1, device).Should().BeApproximately(2.718281828459045d, 1e-6d);
    }

    private static TNumber ExpScalarTest<TNumber>(TNumber number, IDevice device)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var scalar = new Scalar<TNumber>(number);

        scalar.To(device);

        var result = scalar.Exp();

        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVector(IDevice device)
    {
        ExpVectorTest<float>([1, 2, 3], device).Should().BeEquivalentTo(new float[] { 2.718281828459045f, 7.38905609893065f, 20.085536923187668f });
        ExpVectorTest<double>([1, 2, 3], device).Should().BeEquivalentTo(new double[] { 2.718281828459045d, 7.38905609893065d, 20.085536923187668d });
    }

    private static Array ExpVectorTest<TNumber>(TNumber[] numbers, IDevice device)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var vector = Tensor.FromArray<TNumber>(numbers).ToVector();

        vector.To(device);

        var result = vector.Exp();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrix(IDevice device)
    {
        ExpMatrixTest<float>(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device)
            .Should()
            .BeEquivalentTo(new float[,] { { 2.718281828459045f, 7.38905609893065f, 20.085536923187668f }, { 54.598150033144236f, 148.4131591025766f, 403.4287934927351f } });
        ExpMatrixTest<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device)
            .Should()
            .BeEquivalentTo(new double[,] { { 2.718281828459045d, 7.38905609893065d, 20.085536923187668d }, { 54.598150033144236d, 148.4131591025766d, 403.4287934927351d } });
    }

    private static Array ExpMatrixTest<TNumber>(TNumber[,] numbers, IDevice device)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var matrix = Tensor.FromArray<TNumber>(numbers).ToMatrix();

        matrix.To(device);

        var result = matrix.Exp();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensor(IDevice device)
    {
        ExpTensorTest<float>(new float[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device)
            .Should()
            .BeEquivalentTo(
                new float[,,]
                {
                    { { 2.718281828459045f, 7.38905609893065f, 20.085536923187668f }, { 54.598150033144236f, 148.4131591025766f, 403.4287934927351f } },
                    { { 1096.6331584284585f, 2980.9579870417283f, 8103.083927575384f }, { 22026.465794806718f, 59874.14171519782f, 162754.79141900392f } }
                });
        ExpTensorTest<double>(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device)
            .Should()
            .BeEquivalentTo(
                new double[,,]
                {
                    { { 2.718281828459045d, 7.38905609893065d, 20.085536923187668d }, { 54.598150033144236d, 148.4131591025766d, 403.4287934927351d } },
                    { { 1096.6331584284585d, 2980.9579870417283d, 8103.083927575384d }, { 22026.465794806718d, 59874.14171519782d, 162754.79141900392d } }
                });
    }

    private static Array ExpTensorTest<TNumber>(TNumber[,,] numbers, IDevice device)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(numbers);

        tensor.To(device);

        var result = tensor.Exp();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResultAndGradient_GivenMatrix(IDevice device)
    {
        // Arrange
        using var tensor = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, requiresGradient: true);

        tensor.To(device);

        // Act
        using var result = tensor.Exp();

        result.Backward();

        result.To<CpuComputeDevice>();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(new float[,] { { 2.7182817F, 7.389056F }, { 20.085537F, 54.59815F }, { 148.41316F, 403.4288F } }, 1e-4f);
        tensor.Gradient!.Should().NotBeNull();
        tensor.Gradient?.Should().HaveEquivalentElements(new float[,] { { 2.7182817F, 7.389056F }, { 20.085537F, 54.59815F }, { 148.41316F, 403.4288F } });
    }
}