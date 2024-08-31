// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class GetHashCodeShould
{
    [Fact]
    public void GenerateSameHashCode_GivenSameShape()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 4, 5);

        // Act
        var hashCode1 = shape1.GetHashCode();
        var hashCode2 = shape2.GetHashCode();

        // Assert
        hashCode1.Should().Be(hashCode2);
    }

    [Fact]
    public void GenerateSameHashCode_GivenLongElementCount()
    {
        // Arrange
        var shape1 = new Sut(int.MaxValue, int.MaxValue);
        var shape2 = new Sut(int.MaxValue, int.MaxValue);

        // Act
        var hashCode1 = shape1.GetHashCode();
        var hashCode2 = shape2.GetHashCode();

        // Assert
        hashCode1.Should().Be(hashCode2);
    }

    [Fact]
    public void GenerateDifferentHashCode_GivenDifferentShape()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 4, 6);

        // Act
        var hashCode1 = shape1.GetHashCode();
        var hashCode2 = shape2.GetHashCode();

        // Assert
        hashCode1.Should().NotBe(hashCode2);
    }

    [Fact]
    public void GenerateDifferentHashCode_GivenLongUnequalElementCount()
    {
        // Arrange
        var shape1 = new Sut(int.MaxValue, int.MaxValue);
        var shape2 = new Sut(int.MaxValue, int.MaxValue - 1);

        // Act
        var hashCode1 = shape1.GetHashCode();
        var hashCode2 = shape2.GetHashCode();

        // Assert
        hashCode1.Should().NotBe(hashCode2);
    }
}