// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class ConstructorShould
{
    [Fact]
    public void SetPropertiesCorrectly_GivenScalarShape()
    {
        // Arrange & Act
        var shape = new Sut();

        // Assert
        shape.Dimensions.Should().BeEmpty();
        shape.Strides.Should().BeEmpty();
        shape.ElementCount.Should().Be(1);
        shape.Rank.Should().Be(0);
        shape.DataOffset.Should().Be(0);
        shape.IsScalar.Should().BeTrue();
        shape.IsVector.Should().BeFalse();
        shape.IsMatrix.Should().BeFalse();
        shape.IsTensor.Should().BeFalse();
    }

    [Fact]
    public void SetPropertiesCorrectly_GivenVectorShape()
    {
        // Arrange & Act
        var shape = new Sut(3);

        // Assert
        shape.Dimensions.Should().Equal(3);
        shape.Strides.Should().Equal(1);
        shape.ElementCount.Should().Be(3);
        shape.Rank.Should().Be(1);
        shape.DataOffset.Should().Be(0);
        shape.IsScalar.Should().BeFalse();
        shape.IsVector.Should().BeTrue();
        shape.IsMatrix.Should().BeFalse();
        shape.IsTensor.Should().BeFalse();
    }

    [Fact]
    public void SetPropertiesCorrectly_GivenMatrixShape()
    {
        // Arrange & Act
        var shape = new Sut(3, 4);

        // Assert
        shape.Dimensions.Should().Equal(3, 4);
        shape.Strides.Should().Equal(4, 1);
        shape.ElementCount.Should().Be(12);
        shape.Rank.Should().Be(2);
        shape.DataOffset.Should().Be(0);
        shape.IsScalar.Should().BeFalse();
        shape.IsVector.Should().BeFalse();
        shape.IsMatrix.Should().BeTrue();
        shape.IsTensor.Should().BeFalse();
    }

    [Fact]
    public void SetPropertiesCorrectly_GivenTensorShape()
    {
        // Arrange & Act
        var shape = new Sut(3, 4, 5);

        // Assert
        shape.Dimensions.Should().Equal(3, 4, 5);
        shape.Strides.Should().Equal(20, 5, 1);
        shape.ElementCount.Should().Be(60);
        shape.Rank.Should().Be(3);
        shape.DataOffset.Should().Be(0);
        shape.IsScalar.Should().BeFalse();
        shape.IsVector.Should().BeFalse();
        shape.IsMatrix.Should().BeFalse();
        shape.IsTensor.Should().BeTrue();
    }
}