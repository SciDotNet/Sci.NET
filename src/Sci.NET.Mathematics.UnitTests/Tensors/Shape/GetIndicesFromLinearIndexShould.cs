// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class GetIndicesFromLinearIndexShould
{
    [Theory]
    [InlineData(new int[] { 0, 0, 0 }, 0)]
    [InlineData(new int[] { 0, 0, 1 }, 1)]
    [InlineData(new int[] { 0, 1, 0 }, 2)]
    [InlineData(new int[] { 0, 1, 1 }, 3)]
    [InlineData(new int[] { 0, 2, 0 }, 4)]
    [InlineData(new int[] { 0, 2, 1 }, 5)]
    [InlineData(new int[] { 1, 0, 0 }, 6)]
    [InlineData(new int[] { 1, 0, 1 }, 7)]
    [InlineData(new int[] { 1, 1, 0 }, 8)]
    [InlineData(new int[] { 1, 1, 1 }, 9)]
    [InlineData(new int[] { 1, 2, 0 }, 10)]
    [InlineData(new int[] { 1, 2, 1 }, 11)]
    public void ReturnExpectedValues_GivenValidIndices(int[] indices, long expected)
    {
        // Arrange
        var shape = new Sut(2, 3, 2);

        // Act & Assert
        shape.GetLinearIndex(indices).Should().Be(expected);
    }

    [Theory]
    [InlineData(new int[] { 0, 0, 2 })]
    [InlineData(new int[] { 0, 3, 0 })]
    [InlineData(new int[] { 2, 0, 0 })]
    public void ThrowException_GivenInvalidIndices(int[] indices)
    {
        // Arrange
        var shape = new Sut(2, 3, 2);

        // Act
        var action = () => shape.GetLinearIndex(indices);

        // Assert
        action.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(new int[] { 0, 0, 0, 0 })]
    [InlineData(new int[] { 0, 0 })]
    public void ThrowException_GivenInvalidNumberOfIndices(int[] indices)
    {
        // Arrange
        var shape = new Sut(2, 3, 2);

        // Act
        var action = () => shape.GetLinearIndex(indices);

        // Assert
        action.Should().Throw<ArgumentException>();
    }
}