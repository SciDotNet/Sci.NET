// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class InnerProductShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ComputeInnerProduct(IDevice device)
    {
        using var left = Tensor.FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }).WithGradient().ToVector();
        using var right = Tensor.FromArray<float>(new float[] { 8, 7, 6, 5, 4, 3, 2, 1 }).WithGradient().ToVector();

        left.To(device);
        right.To(device);

        using var result = left.Inner(right);

        result.Backward();

        result.Value.Should().Be(120);

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(new float[] { 8, 7, 6, 5, 4, 3, 2, 1 }, 1e-6f);

        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_GivenFp32(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }).ToVector();
        var right = Tensor.FromArray<float>(new float[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }).ToVector();
        const float expected = 220;

        left.To(device);
        right.To(device);

        // Act
        var result = left.Inner(right);

        // Assert
        result.Memory.ToSystemMemory()[0].Should().Be(expected);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_GivenFp64(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }).ToVector();
        var right = Tensor.FromArray<double>(new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }).ToVector();
        const double expected = 220;

        left.To(device);
        right.To(device);

        // Act
        var result = left.Inner(right);

        // Assert
        result.Memory.ToSystemMemory()[0].Should().Be(expected);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_GivenLargeFp32(IDevice device)
    {
        // Arrange
        var left = Tensor.Random.Uniform<float>(new Shape(50000), -1.0f, 1.0f).ToVector();
        var right = Tensor.Random.Uniform<float>(new Shape(50000), -1.0f, 1.0f).ToVector();

        var expected = 0.0f;

        for (var i = 0; i < left.Length; i++)
        {
            expected += left[i].Value * right[i].Value;
        }

        left.To(device);
        right.To(device);

        // Act
        var result = left.Inner(right);

        // Assert
        result.Memory.ToSystemMemory()[0].Should().BeApproximately(expected, 0.01f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_GivenLargeFp64(IDevice device)
    {
        // Arrange
        var left = Tensor.Random.Uniform<double>(new Shape(50000), -1.0f, 1.0f).ToVector();
        var right = Tensor.Random.Uniform<double>(new Shape(50000), -1.0f, 1.0f).ToVector();

        var expected = 0.0d;

        for (var i = 0; i < left.Length; i++)
        {
            expected += left[i].Value * right[i].Value;
        }

        left.To(device);
        right.To(device);

        // Act
        var result = left.Inner(right);

        // Assert
        result.Memory.ToSystemMemory()[0].Should().BeApproximately(expected, 0.01f);
    }
}