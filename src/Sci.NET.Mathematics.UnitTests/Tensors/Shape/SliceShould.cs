// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class SliceShould
{
    [Theory]
    [InlineData(new int[] { 3, 3, 3, 3 }, new int[] { 1, 0, 1, 0 }, new int[0], 30)]
    [InlineData(new int[] { 3, 3, 3, 3 }, new int[] { 1, 0, 1 }, new int[] { 3 }, 30)]
    [InlineData(new int[] { 3, 4, 3, 3 }, new int[] { 1, 0, 1 }, new int[] { 3 }, 39)]
    [InlineData(new int[] { 3, 4, 3, 3 }, new int[] { 1, 0 }, new int[] { 3, 3 }, 36)]
    [InlineData(new int[] { 3, 4, 3, 3 }, new int[] { 1 }, new int[] { 4, 3, 3 }, 36)]
    [InlineData(new int[] { 3, 4, 3, 3 }, new int[] { }, new int[] { 3, 4, 3, 3 }, 0)]
    public void ReturnCorrectValue_GivenScalarShape(
        int[] dimensions,
        int[] indices,
        int[] resultShape,
        long resultOffset)
    {
        // Arrange
        var shape = new Sut(dimensions);

        // Act
        var slice = shape.Slice(indices);

        // Assert
        slice.Dimensions.Should().BeEquivalentTo(resultShape);
        slice.DataOffset.Should().Be(resultOffset);
    }

    [Fact]
    public void ThrowException_GivenTooManyIndices()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act
        var act = () => shape.Slice(1, 1, 1, 1);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData(new int[] { 5 })]
    [InlineData(new int[] { -1 })]
    public void ThrowException_GivenInvalidIndices(int[] indices)
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act
        var act = () => shape.Slice(indices);

        // Assert
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}