// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Tensors;

public partial class TensorTests
{
    [Theory]
    [InlineData(new int[] { 2, 3 }, new int[] { 0, 0 }, 0)]
    [InlineData(new int[] { 2, 3 }, new int[] { 0, 1 }, 1)]
    [InlineData(new int[] { 2, 3 }, new int[] { 0, 2 }, 2)]
    [InlineData(new int[] { 2, 3 }, new int[] { 1, 0 }, 3)]
    [InlineData(new int[] { 2, 3 }, new int[] { 1, 1 }, 4)]
    [InlineData(new int[] { 2, 3 }, new int[] { 1, 2 }, 5)]
    public void GetIndices_GivenValidIndices_ReturnsExpectedResult(int[] shape, int[] indices, long index)
    {
        // Arrange
        var tensor = new Tensor<int>(new Shape(shape));

        // Act
        var result = tensor.GetIndicesFromLinearIndex(index);

        // Assert
        result.Should().BeEquivalentTo(indices);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(6)]
    public void GetIndices_GivenInvalidLinearIndex_ThrowsException(int index)
    {
        // Arrange
        var tensor = new Tensor<int>(new Shape(2, 3));

        // Act
        var act = () => tensor.GetIndicesFromLinearIndex(index);

        // Assert
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}