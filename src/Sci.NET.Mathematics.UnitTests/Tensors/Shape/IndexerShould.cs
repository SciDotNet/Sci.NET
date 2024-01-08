// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class IndexerShould
{
    [Fact]
    public void ReturnCorrectValue_GivenScalarShape()
    {
        // Arrange
        var shape = new Sut();

        // Act & Assert
        shape[0].Should().Be(1);
    }

    [Fact]
    public void ReturnCorrectValue_GivenVectorShape()
    {
        // Arrange
        var shape = new Sut(3);

        // Act & Assert
        shape[0].Should().Be(3);
    }

    [Fact]
    public void ReturnCorrectValue_GivenMatrixShape()
    {
        // Arrange
        var shape = new Sut(3, 4);

        // Act & Assert
        shape[0].Should().Be(3);
        shape[1].Should().Be(4);
    }

    [Fact]
    public void ReturnCorrectValue_GivenTensorShape()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act & Assert
        shape[0].Should().Be(3);
        shape[1].Should().Be(4);
        shape[2].Should().Be(5);
    }

    [Fact]
    public void ReturnCorrectValue_GivenRange()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act & Assert
        shape[0..2].Should().BeEquivalentTo(new int[] { 3, 4 });
        shape[1..3].Should().BeEquivalentTo(new int[] { 4, 5 });
    }
}