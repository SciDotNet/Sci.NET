// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class ToStringShould
{
    [Fact]
    public void ReturnCorrectValue_GivenScalarShape()
    {
        // Arrange
        var shape = new Sut();

        // Act & Assert
        shape.ToString().Should().Be("[]");
    }

    [Fact]
    public void ReturnCorrectValue_GivenVectorShape()
    {
        // Arrange
        var shape = new Sut(3);

        // Act & Assert
        shape.ToString().Should().Be("[3]");
    }

    [Fact]
    public void ReturnCorrectValue_GivenMatrixShape()
    {
        // Arrange
        var shape = new Sut(3, 4);

        // Act & Assert
        shape.ToString().Should().Be("[3, 4]");
    }

    [Fact]
    public void ReturnCorrectValue_GivenTensorShape()
    {
        // Arrange
        var shape = new Sut(3, 4, 5);

        // Act & Assert
        shape.ToString().Should().Be("[3, 4, 5]");
    }
}