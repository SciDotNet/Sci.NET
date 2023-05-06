// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class GetIndicesFromLinearIndex
{
    [Theory]
    [InlineData(0, new int[] { 0, 0, 0 })]
    [InlineData(1, new int[] { 0, 0, 1 })]
    [InlineData(2, new int[] { 0, 1, 0 })]
    [InlineData(3, new int[] { 0, 1, 1 })]
    [InlineData(4, new int[] { 0, 2, 0 })]
    [InlineData(5, new int[] { 0, 2, 1 })]
    [InlineData(6, new int[] { 1, 0, 0 })]
    [InlineData(7, new int[] { 1, 0, 1 })]
    [InlineData(8, new int[] { 1, 1, 0 })]
    [InlineData(9, new int[] { 1, 1, 1 })]
    [InlineData(10, new int[] { 1, 2, 0 })]
    [InlineData(11, new int[] { 1, 2, 1 })]
    public void GetIndicesFromLinearIndex_GivenValidIndices(long linearIndex, int[] expectedIndices)
    {
        // Arrange
        var shape = new Sut(2, 3, 2);

        // Act & Assert
        shape.GetIndicesFromLinearIndex(linearIndex)
            .Should()
            .BeEquivalentTo(expectedIndices);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(12)]
    public void ThrowException_GivenInvalidIndex(long linearIndex)
    {
        // Arrange
        var shape = new Sut(2, 3, 2);

        // Act
        var action = () => shape.GetIndicesFromLinearIndex(linearIndex);

        // Assert
        action.Should().Throw<ArgumentOutOfRangeException>();
    }
}