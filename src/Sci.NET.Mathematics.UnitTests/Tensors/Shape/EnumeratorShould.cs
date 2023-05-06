// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class EnumeratorShould
{
    [Fact]
    public void ReturnCorrectValue_GivenScalarShape()
    {
        // Arrange
        var shape = new Sut();
        var expected = Array.Empty<int>();

        // Act & Assert
        shape.GetEnumerator().Should().BeEquivalentTo(expected);
    }

    [Fact]
    public void ReturnCorrectValue_GivenVectorShape()
    {
        // Arrange
        var shape = new Sut(3);

        // Act
        using var enumerator = shape.GetEnumerator();

        // Assert
        enumerator.MoveNext().Should().BeTrue();
        enumerator.Current.Should().Be(3);
        enumerator.MoveNext().Should().BeFalse();
    }

    [Fact]
    public void ReturnCorrectValue_GivenMatrixShape()
    {
        // Arrange
        var shape = new Sut(3, 4);

        // Act
        using var enumerator = shape.GetEnumerator();

        // Assert
        enumerator.MoveNext().Should().BeTrue();
        enumerator.Current.Should().Be(3);
        enumerator.MoveNext().Should().BeTrue();
        enumerator.Current.Should().Be(4);
        enumerator.MoveNext().Should().BeFalse();
    }

    [Fact]
    public void ReturnCorrectValue_GivenTensorShape()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act
        using var enumerator = shape.GetEnumerator();

        // Assert
        enumerator.MoveNext().Should().BeTrue();
        enumerator.Current.Should().Be(3);
        enumerator.MoveNext().Should().BeTrue();
        enumerator.Current.Should().Be(4);
        enumerator.MoveNext().Should().BeTrue();
        enumerator.Current.Should().Be(5);
        enumerator.MoveNext().Should().BeFalse();
    }

    [Fact]
    public void GetEnumerator_IteratesCorrectly()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);
        var expected = new int[] { 3, 4, 5 };
        var actual = new List<int>();

        // Act
        foreach (var dimension in shape)
        {
#pragma warning disable RCS1235
            actual.Add(dimension);
#pragma warning restore RCS1235
        }

        // Assert
        actual.Should().BeEquivalentTo(expected);
    }
}