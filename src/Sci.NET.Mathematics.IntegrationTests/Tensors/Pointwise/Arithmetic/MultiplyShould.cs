// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public class MultiplyShould : IArithmeticOperatorTests
{
    [Fact]
    public void ReturnExpectedResult_GivenScalarScalar()
    {
        // Arrange
        using var left = new Scalar<int>(2);
        using var right = new Scalar<int>(2);
        const int expectedValue = 4;

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 4, 8, 16, 32 };

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 4, 8, 16, 32 };

        // Act
        var actual = left.Multiply(right);

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
        using var left = new Scalar<int>(64);
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 4, 8, 16, 32 };

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 1, 4, 16, 64 };

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 1, 4, 16, 4 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 1, 4, 16, 64, 16, 64, 256, 4 };

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 4, 8, 16, 32 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        var expectedValues = new int[] { 1, 4, 16, 4 };

        // Act
        var actual = left.Multiply(right);

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
        var expectedValues = new int[] { 1, 4, 16, 64 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 1, 4, 16, 64, 16, 64, 256, 1024 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { 4, 8, 16, 32, 64, 128, 256, 512 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        var expectedValues = new int[] { 1, 4, 16, 64, 16, 64, 256, 4 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();
        var expectedValues = new int[] { 1, 4, 16, 64, 16, 64, 256, 1024 };

        // Act
        var actual = left.Multiply(right);

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
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 1, 4, 16, 64, 256, 1024, 4096, 16384 };

        // Act
        var actual = left.Multiply(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }
}