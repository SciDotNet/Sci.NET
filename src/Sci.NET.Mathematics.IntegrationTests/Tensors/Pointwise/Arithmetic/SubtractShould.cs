// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public class SubtractShould : IntegrationTestBase, IArithmeticOperatorTests
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarScalar(IDevice device)
    {
        // Arrange
        using var left = new Scalar<int>(2);
        using var right = new Scalar<int>(2);
        const int expectedValue = 0;

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .Value.Should()
            .Be(expectedValue);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarVector(IDevice device)
    {
        // Arrange
        using var left = new Scalar<int>(4);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        var expectedValues = new int[] { 3, 2, 0, -4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarMatrix(IDevice device)
    {
        // Arrange
        using var left = new Scalar<int>(4);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        var expectedValues = new int[] { 3, 2, 0, -4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarTensor(IDevice device)
    {
        // Arrange
        using var left = new Scalar<int>(4);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2, 1)
            .ToTensor();
        var expectedValues = new int[] { 3, 2, 0, -4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorScalar(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { -3, -2, 0, 4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorVector(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        var expectedValues = new int[] { 0, 0, 0, 0 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorMatrix(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        var expectedValues = new int[] { 0, 0, -3, -6 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorTensor(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(1, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 0, 0, -3, -6 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixScalar(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { -3, -2, 0, 4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixVector(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 4, 8, 12, 16 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        var expectedValues = new int[] { 0, 0, 6, 3 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixMatrix(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        var expectedValues = new int[] { 0, 0, 0, 0 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixTensor(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 4, 8, 12, 16 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(1, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 0, 2, 3, -4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorScalar(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(1, 2, 2)
            .ToTensor();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { -3, -2, 0, 4 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorVector(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 4, 8, 12, 16 })
            .Reshape(1, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        var expectedValues = new int[] { 3, 6, 11, 14 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorMatrix(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 4, 8, 12, 16 })
            .Reshape(1, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        var expectedValues = new int[] { 3, 6, 11, 14 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorTensor(IDevice device)
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(1, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(1, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 0, 0, 0, 0 };

        left.To(device);
        right.To(device);

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }
}