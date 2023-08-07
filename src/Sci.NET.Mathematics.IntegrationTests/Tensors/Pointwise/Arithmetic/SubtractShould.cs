// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public class SubtractShould : IArithmeticOperatorTests
{
    [Fact]
    public void ReturnExpectedResult_GivenScalarScalar()
    {
        // Arrange
        using var left = new Scalar<int>(2);
        using var right = new Scalar<int>(2);
        const int expectedValue = 0;

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .Value.Should()
            .Be(expectedValue);
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarVector()
    {
        // Arrange
        using var left = new Scalar<int>(4);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        var expectedValues = new int[] { 3, 2, 0, -4 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarMatrix()
    {
        // Arrange
        using var left = new Scalar<int>(4);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        var expectedValues = new int[] { 3, 2, 0, -4 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarTensor()
    {
        // Arrange
        using var left = new Scalar<int>(4);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2, 1)
            .ToTensor();
        var expectedValues = new int[] { 3, 2, 0, -4 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorScalar()
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { -3, -2, 0, 4 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorVector()
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        var expectedValues = new int[] { 0, 0, 0, 0 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorMatrix()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorTensor()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixScalar()
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { -3, -2, 0, 4 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixVector()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixMatrix()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixTensor()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenTensorScalar()
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(1, 2, 2)
            .ToTensor();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { -3, -2, 0, 4 };

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenTensorVector()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenTensorMatrix()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }

    [Fact]
    public void ReturnExpectedResult_GivenTensorTensor()
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

        // Act
        var actual = left.Subtract(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }
}