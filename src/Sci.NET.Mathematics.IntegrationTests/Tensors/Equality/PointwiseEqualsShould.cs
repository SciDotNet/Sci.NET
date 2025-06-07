﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Equality;

public class PointwiseEqualsShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallFloat32Tensors(IDevice device)
    {
        // Arrange
        using var left = Tensor.FromArray<float>(Enumerable.Range(0, 50).Select(x => (float)x).ToArray()).Reshape(5, 10).WithGradient();
        using var right = Tensor.FromArray<float>(Enumerable.Range(0, 50).Select(x => (float)x).ToArray()).Reshape(5, 10).WithGradient();
        using var expectedResult = Tensor.Ones<float>(5, 10);

        left.Memory[8] = 0.0f;
        expectedResult.Memory[8] = 0.0f;
        left.Memory[25] = 0.0f;
        expectedResult.Memory[25] = 0.0f;

        left.To(device);
        right.To(device);

        // Act
        var result = left.PointwiseEquals(right);

        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), 1e-6f);

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<float>(5, 10).ToArray(), 1e-6f);
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<float>(5, 10).ToArray(), 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeFloat32Tensors(IDevice device)
    {
        // Arrange
        using var left = Tensor.FromArray<float>(Enumerable.Range(0, 5000).Select(x => (float)x).ToArray()).Reshape(50, 100).WithGradient();
        using var right = Tensor.FromArray<float>(Enumerable.Range(0, 5000).Select(x => (float)x).ToArray()).Reshape(50, 100).WithGradient();
        using var expectedResult = Tensor.Ones<float>(50, 100);

        left.Memory[8] = 0.0f;
        expectedResult.Memory[8] = 0.0f;
        left.Memory[25] = 0.0f;
        expectedResult.Memory[25] = 0.0f;

        left.To(device);
        right.To(device);

        // Act
        var result = left.PointwiseEquals(right);

        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), 1e-6f);

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<float>(50, 100).ToArray(), 1e-6f);
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<float>(50, 100).ToArray(), 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallFloat64Tensors(IDevice device)
    {
        // Arrange
        using var left = Tensor.FromArray<double>(Enumerable.Range(0, 50).Select(x => (double)x).ToArray()).Reshape(5, 10).WithGradient();
        using var right = Tensor.FromArray<double>(Enumerable.Range(0, 50).Select(x => (double)x).ToArray()).Reshape(5, 10).WithGradient();
        using var expectedResult = Tensor.Ones<double>(5, 10);

        left.Memory[8] = 0.0f;
        expectedResult.Memory[8] = 0.0f;
        left.Memory[25] = 0.0f;
        expectedResult.Memory[25] = 0.0f;

        left.To(device);
        right.To(device);

        // Act
        var result = left.PointwiseEquals(right);

        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), 1e-6f);

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<double>(5, 10).ToArray(), 1e-6f);
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<double>(5, 10).ToArray(), 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeFloat64Tensors(IDevice device)
    {
        // Arrange
        using var left = Tensor.FromArray<double>(Enumerable.Range(0, 5000).Select(x => (double)x).ToArray()).Reshape(50, 100).WithGradient();
        using var right = Tensor.FromArray<double>(Enumerable.Range(0, 5000).Select(x => (double)x).ToArray()).Reshape(50, 100).WithGradient();
        using var expectedResult = Tensor.Ones<double>(50, 100);

        left.Memory[8] = 0.0f;
        expectedResult.Memory[8] = 0.0f;
        left.Memory[25] = 0.0f;
        expectedResult.Memory[25] = 0.0f;

        left.To(device);
        right.To(device);

        // Act
        var result = left.PointwiseEquals(right);

        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), 1e-6f);

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<double>(50, 100).ToArray(), 1e-6f);
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(Tensor.Ones<double>(50, 100).ToArray(), 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSmallInt32Tensors(IDevice device)
    {
        // Arrange
        using var left = Tensor.FromArray<int>(Enumerable.Range(0, 50).ToArray()).Reshape(5, 10).WithGradient();
        using var right = Tensor.FromArray<int>(Enumerable.Range(0, 50).ToArray()).Reshape(5, 10).WithGradient();
        using var expectedResult = Tensor.Ones<int>(5, 10);

        left.Memory[8] = 0;
        expectedResult.Memory[8] = 0;
        left.Memory[25] = 0;
        expectedResult.Memory[25] = 0;

        left.To(device);
        right.To(device);

        // Act
        var result = left.PointwiseEquals(right);

        result.Backward();

        // Assert
        result.Should().HaveEquivalentElements(expectedResult.ToArray());

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveEquivalentElements(Tensor.Ones<int>(5, 10).ToArray());
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveEquivalentElements(Tensor.Ones<int>(5, 10).ToArray());
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeInt32Tensors(IDevice device)
    {
        // Arrange
        using var left = Tensor.FromArray<int>(Enumerable.Range(0, 5000).ToArray()).Reshape(50, 100).WithGradient();
        using var right = Tensor.FromArray<int>(Enumerable.Range(0, 5000).ToArray()).Reshape(50, 100).WithGradient();
        using var expectedResult = Tensor.Ones<int>(50, 100);

        left.Memory[8] = 0;
        expectedResult.Memory[8] = 0;
        left.Memory[25] = 0;
        expectedResult.Memory[25] = 0;

        left.To(device);
        right.To(device);

        // Act
        var result = left.PointwiseEquals(right);

        result.Backward();

        // Assert
        result.Should().HaveEquivalentElements(expectedResult.ToArray());

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveEquivalentElements(Tensor.Ones<int>(50, 100).ToArray());
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveEquivalentElements(Tensor.Ones<int>(50, 100).ToArray());
    }
}