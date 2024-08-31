// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class EqualsShould
{
    [Fact]
    public void ReturnTrue_GivenEqualShapes()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 4, 5);

        // Act & Assert
        shape1.Equals(shape2).Should().BeTrue();
    }

    [Fact]
    public void ReturnFalse_GivenUnequalShapes()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 4, 6);

        // Act & Assert
        shape1.Equals(shape2).Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenEqualElementCountDifferentDimensionCount()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 20);

        // Act & Assert
        shape1.Equals(shape2).Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenEqualElementCount_ReorderedDimensions()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(4, 5, 3);

        // Act & Assert
        shape1.Equals(shape2).Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenOtherObject()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act & Assert
        shape.Equals(new object()).Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenNull()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act & Assert
#pragma warning disable CA1508
        shape.Equals(null).Should().BeFalse();
#pragma warning restore CA1508
    }
}