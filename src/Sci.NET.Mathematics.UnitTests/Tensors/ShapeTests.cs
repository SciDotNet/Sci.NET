// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Tensors;

public class ShapeTests
{
    [Fact]
    public void Ctor_GivenEmptyDimensions_SetsExpectedValues()
    {
        // Arrange & Act
        var shape = new Shape();

        // Assert
        shape.Dimensions.Should().BeEmpty();
        shape.Rank.Should().Be(0);
        shape.Strides.Should().BeEmpty();
        shape.ElementCount.Should().Be(1);
        shape.IsScalar.Should().BeTrue();
        shape.IsVector.Should().BeFalse();
        shape.IsMatrix.Should().BeFalse();
    }

    [Fact]
    public void Ctor_GivenOneDimension_SetsExpectedValues()
    {
        // Arrange & Act
        var shape = new Shape(2);

        // Assert
        shape.Dimensions.Should().Equal(2);
        shape.Rank.Should().Be(1);
        shape.Strides.Should().Equal(1);
        shape.ElementCount.Should().Be(2);
        shape.IsScalar.Should().BeFalse();
        shape.IsVector.Should().BeTrue();
        shape.IsMatrix.Should().BeFalse();
    }

    [Fact]
    public void Ctor_GivenTwoDimensions_SetsExpectedValues()
    {
        // Arrange & Act
        var shape = new Shape(2, 3);

        // Assert
        shape.Dimensions.Should().Equal(2, 3);
        shape.Rank.Should().Be(2);
        shape.Strides.Should().Equal(3, 1);
        shape.ElementCount.Should().Be(6);
        shape.IsScalar.Should().BeFalse();
        shape.IsVector.Should().BeFalse();
        shape.IsMatrix.Should().BeTrue();
    }

    [Fact]
    public void Ctor_GivenThreeDimensions_SetsExpectedValues()
    {
        // Arrange & Act
        var shape = new Shape(2, 3, 4);

        // Assert
        shape.Dimensions.Should().Equal(2, 3, 4);
        shape.Rank.Should().Be(3);
        shape.Strides.Should().Equal(12, 4, 1);
        shape.ElementCount.Should().Be(24);
    }

    [Theory]
    [InlineData(0, 2)]
    [InlineData(1, 3)]
    [InlineData(2, 4)]
    public void Indexer_GivenValidIndex_ReturnsExpectedValue(int index, int value)
    {
        // Arrange
        var shape = new Shape(2, 3, 4);

        // Act
        var result = shape[index];

        // Assert
        result.Should().Be(value);
    }

    [Theory]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 4 }, true)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 5 }, false)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3 }, false)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 4, 5 }, false)]
    public void EqualsOperator_GivenDifferentShapes_ReturnsExpectedResult(int[] shape1, int[] shape2, bool expected)
    {
        // Arrange
        var first = new Shape(shape1);
        var second = new Shape(shape2);

        // Act
        var result = first == second;

        // Assert
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 4 }, false)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 5 }, true)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3 }, true)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 4, 5 }, true)]
    public void NotEqualsOperator_GivenDifferentShapes_ReturnsExpectedResult(int[] shape1, int[] shape2, bool expected)
    {
        // Arrange
        var first = new Shape(shape1);
        var second = new Shape(shape2);

        // Act
        var result = first != second;

        // Assert
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(0, new int[] { 0, 0, 0 })]
    [InlineData(1, new int[] { 0, 0, 1 })]
    [InlineData(2, new int[] { 0, 0, 2 })]
    [InlineData(3, new int[] { 0, 0, 3 })]
    [InlineData(4, new int[] { 0, 1, 0 })]
    [InlineData(5, new int[] { 0, 1, 1 })]
    [InlineData(6, new int[] { 0, 1, 2 })]
    [InlineData(7, new int[] { 0, 1, 3 })]
    [InlineData(8, new int[] { 0, 2, 0 })]
    [InlineData(9, new int[] { 0, 2, 1 })]
    [InlineData(10, new int[] { 0, 2, 2 })]
    [InlineData(11, new int[] { 0, 2, 3 })]
    [InlineData(12, new int[] { 1, 0, 0 })]
    [InlineData(13, new int[] { 1, 0, 1 })]
    [InlineData(14, new int[] { 1, 0, 2 })]
    [InlineData(15, new int[] { 1, 0, 3 })]
    [InlineData(16, new int[] { 1, 1, 0 })]
    [InlineData(17, new int[] { 1, 1, 1 })]
    [InlineData(18, new int[] { 1, 1, 2 })]
    [InlineData(19, new int[] { 1, 1, 3 })]
    [InlineData(20, new int[] { 1, 2, 0 })]
    [InlineData(21, new int[] { 1, 2, 1 })]
    [InlineData(22, new int[] { 1, 2, 2 })]
    [InlineData(23, new int[] { 1, 2, 3 })]
    public void GetIndicesFromLinearIndex(int index, int[] expectedIndices)
    {
        // Arrange
        var shape = new Shape(2, 3, 4);

        // Act
        var result = shape.GetIndicesFromLinearIndex(index);

        // Assert
        result.Should().Equal(expectedIndices);
    }

    [Theory]
    [InlineData(new int[] { 0, 0, 0 }, 0)]
    [InlineData(new int[] { 0, 0, 1 }, 1)]
    [InlineData(new int[] { 0, 0, 2 }, 2)]
    [InlineData(new int[] { 0, 0, 3 }, 3)]
    [InlineData(new int[] { 0, 1, 0 }, 4)]
    [InlineData(new int[] { 0, 1, 1 }, 5)]
    [InlineData(new int[] { 0, 1, 2 }, 6)]
    [InlineData(new int[] { 0, 1, 3 }, 7)]
    [InlineData(new int[] { 0, 2, 0 }, 8)]
    [InlineData(new int[] { 0, 2, 1 }, 9)]
    [InlineData(new int[] { 0, 2, 2 }, 10)]
    [InlineData(new int[] { 0, 2, 3 }, 11)]
    [InlineData(new int[] { 1, 0, 0 }, 12)]
    [InlineData(new int[] { 1, 0, 1 }, 13)]
    [InlineData(new int[] { 1, 0, 2 }, 14)]
    [InlineData(new int[] { 1, 0, 3 }, 15)]
    [InlineData(new int[] { 1, 1, 0 }, 16)]
    [InlineData(new int[] { 1, 1, 1 }, 17)]
    [InlineData(new int[] { 1, 1, 2 }, 18)]
    [InlineData(new int[] { 1, 1, 3 }, 19)]
    [InlineData(new int[] { 1, 2, 0 }, 20)]
    [InlineData(new int[] { 1, 2, 1 }, 21)]
    [InlineData(new int[] { 1, 2, 2 }, 22)]
    [InlineData(new int[] { 1, 2, 3 }, 23)]
    public void GetLinearIndexFromIndices(int[] indices, int expectedIndex)
    {
        // Arrange
        var shape = new Shape(2, 3, 4);

        // Act
        var result = shape.GetLinearIndex(indices);

        // Assert
        result.Should().Be(expectedIndex);
    }

    [Theory]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 4 }, true)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 5 }, false)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3 }, false)]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 3, 4, 5 }, false)]
    public void Equals_GivenDifferentShapes_ReturnsExpectedResult(int[] shape1, int[] shape2, bool expected)
    {
        // Arrange
        var first = new Shape(shape1);
        var second = new Shape(shape2);

        // Act
        var result = first.Equals(second);

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void Equals_GivenNull_ReturnsFalse()
    {
        // Arrange
        var shape = new Shape(2, 3, 4);

        // Act
#pragma warning disable CA1508
        var result = shape.Equals(null);
#pragma warning restore CA1508

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void Equals_GivenObject_ReturnsFalse()
    {
        // Arrange
        var shape = new Shape(2, 3, 4);

        // Act
        var result = shape.Equals(new object());

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void GetHashCode_GivenObjectOfSameType_ReturnsTrue()
    {
        // Arrange
        var first = new Shape(2, 3, 4);
        var second = new Shape(2, 3, 4);

        // Act
        var result = first.GetHashCode() == second.GetHashCode();

        // Assert
        result.Should().BeTrue();
    }

    [Fact]
    public void ToString_GivenShape_ReturnsExpectedResult()
    {
        // Arrange
        var shape = new Shape(2, 3, 4);

        // Act
        var result = shape.ToString();

        // Assert
        result.Should().Be("Shape<2, 3, 4>");
    }
}