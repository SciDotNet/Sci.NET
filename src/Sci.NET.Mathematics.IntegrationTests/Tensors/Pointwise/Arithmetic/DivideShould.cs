// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public class DivideShould : IArithmeticOperatorTests
{
    [Fact]
    public void ReturnExpectedResult_GivenScalarScalar()
    {
        // Arrange
        using var left = new Scalar<int>(2);
        using var right = new Scalar<int>(2);
        const int expectedValue = 1;

        // Act
        var actual = left.Divide(right);

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
        var expectedValues = new int[] { 4, 2, 1, 0 };

        // Act
        var actual = left.Divide(right);

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
        var expectedValues = new int[] { 4, 2, 1, 0 };

        // Act
        var actual = left.Divide(right);

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
            .Reshape(2, 2, 2);
        var expectedValues = new int[] { 64, 32, 16, 8, 4, 2, 1, 0 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .ToVector();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { 0, 1, 2, 4 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .ToVector();
        var expectedValues = new int[] { 2, 2, 2, 2 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 256, 128, 64, 32 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 2, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 4)
            .ToMatrix();
        var expectedValues = new int[] { 128, 64, 16, 4, 16, 4, 1, 0 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 256, 128 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 256, 64, 64, 16, 16, 4, 4, 1 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { 0, 1, 2, 4 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 2, 4 })
            .ToVector();
        var expectedValues = new int[] { 1, 1, 4, 4 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();

        // Act
        var actual = left.Divide(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(new int[] { 2, 2, 2, 2 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixTensor()
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 256, 128, 64, 32 })
            .Reshape(2, 2)
            .ToMatrix();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8, 16, 32, 64, 128 })
            .Reshape(2, 2, 2)
            .ToTensor();
        var expectedValues = new int[] { 256, 64, 16, 4, 16, 4, 1, 0 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .Reshape(2, 2, 1)
            .ToTensor();
        using var right = new Scalar<int>(4);
        var expectedValues = new int[] { 0, 1, 2, 4 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16, 32, 64, 128, 256 })
            .Reshape(2, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2 })
            .ToVector();
        var expectedValues = new int[] { 2, 2, 8, 8, 32, 32, 128, 128 };

        // Act
        var actual = left.Divide(right);

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
            .FromArray<int>(new int[] { 2, 4, 8, 16, 32, 64, 128, 256 })
            .Reshape(2, 2, 2)
            .ToTensor();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 4, 8 })
            .Reshape(2, 2)
            .ToMatrix();

        // Act
        var actual = left.Divide(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(new int[] { 2, 2, 8, 8, 32, 32, 128, 128 });
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
            .FromArray<int>(new int[] { 2, 4, 8, 16 })
            .Reshape(1, 2, 2)
            .ToTensor();

        // Act
        var actual = left.Divide(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(new int[] { 0, 0, 2, 2, 8, 8, 32, 32 });
    }
}