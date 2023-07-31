// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Manipulation.Broadcasting;

public class BroadcastShould
{
    [Fact]
    public void ReturnExpectedResult_GivenScalarAndScalar()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = new Scalar<int>(2);
        using var expected = new Scalar<int>(1);

        // Act
        var actual = left.Broadcast(right.Shape).ToScalar();

        // Assert
        actual.Value.Should().Be(expected.Value);
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndVector()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var expected = Tensor.FromArray<int>(new int[] { 1, 1, 1, 1 }).ToVector();

        // Act
        var actual = left.Broadcast(right.Shape).ToVector();

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndMatrix()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var expected = Tensor.FromArray<int>(new int[] { 1, 1, 1, 1, 1, 1 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Broadcast(right.Shape).ToMatrix();

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndTensor()
    {
        // Arrange
        using var left = new Scalar<int>(1);
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3, 1);
        using var expected = Tensor.FromArray<int>(new int[] { 1, 1, 1, 1, 1, 1 }).Reshape(2, 3, 1);

        // Act
        var actual = left.Broadcast(right.Shape);

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndScalar()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var right = new Scalar<int>(1);
        using var expected = Tensor.FromArray<int>(new int[] { 1, 1, 1, 1 }).ToVector();

        // Act
        var actual = right.Broadcast(left.Shape).ToVector();

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndVector()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
        using var expected = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();

        // Act
        var actual = left.Broadcast(right.Shape).ToVector();

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }

    [Fact]
    public void ReturnExpectedResult_GivenVectorAndMatrix()
    {
        // Arrange
        using var left = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        using var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6 }).Reshape(2, 3).ToMatrix();
        using var expected = Tensor.FromArray<int>(new int[] { 1, 2, 3, 1, 2, 3 }).Reshape(2, 3).ToMatrix();

        // Act
        var actual = left.Broadcast(right.Shape).ToMatrix();

        // Assert
        actual.ToArray().Should().BeEquivalentTo(expected.ToArray());
    }
}