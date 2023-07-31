// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public class AddShould
{
    [Fact]
    public void ReturnExpectedResult_GivenScalarAndScalar()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = new Scalar<int>(2);
        using var expected = new Scalar<int>(3);

        // Act
        var actual = left.Add(right);

        // Assert
        actual.Value.Should().Be(expected.Value);
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndVector()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5 }).ToVector();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndMatrix()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5, 6, 7 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndTensor()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3, 1);
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5, 6, 7 }).Reshape(2, 3, 1);

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndScalar()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var right = new Scalar<int>(1);
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5 }).ToVector();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndVector()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 4, 6, 8 }).ToVector();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndMatrix()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2 }).ToVector();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 6, 7, 8 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenBroadcastableVectorAndMatrix()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 4, 6, 5, 7, 9 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndTensor()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2 }).ToVector();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3, 1);
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5, 6, 7 }).Reshape(2, 3, 1);

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixAndScalar()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var right = new Scalar<int>(1);
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5, 6, 7 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixAndVector()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2 }).ToVector();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 6, 7, 8 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixAndMatrix()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var expected = Tensor.FromArray<int>(new int[] { 2, 4, 6, 8, 10, 12 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrixAndTensor()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).Reshape(1, 3).ToMatrix();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3, 1);
        using var expected = Tensor.FromArray<int>(new int[] { 2, 4, 6, 5, 7, 9 }).Reshape(2, 3, 1);

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenTensorAndScalar()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3, 1);
        using var right = new Scalar<int>(1);
        using var expected = Tensor.FromArray<int>(new int[] { 2, 3, 4, 5, 6, 7 }).Reshape(2, 3, 1);

        // Act
        var actual = left.Add(right);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }
}