// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Tensors;

public partial class TensorTests
{
    [Fact]
    public void Ctor_GivenShapeAndValues_SetsValuesCorrectly()
    {
        // Arrange
        var shape = new Shape(2, 3);

        // Act
        var tensor = new Tensor<int>(shape);

        // Assert
        tensor.Rank.Should().Be(2);
        tensor.ElementCount.Should().Be(6);
        tensor.IsScalar.Should().BeFalse();
        tensor.IsVector.Should().BeFalse();
        tensor.IsMatrix.Should().BeTrue();
        tensor.ReferenceCount.GetCount().Should().Be(1);

        tensor.Strides.Should().BeEquivalentTo(
            new int[]
            {
                3, 1
            });

        tensor.Dimensions.Should().BeEquivalentTo(
            new int[]
            {
                2, 3
            });

        tensor.Data.ToArray().Should().AllBeEquivalentTo(0);
    }

    [Fact]
    public void Ctor_GivenShapeAndMemoryBlock_SetsCorrectValues()
    {
        // Arrange
        var shape = new Shape(2, 5);
        var memoryBlock = new SystemMemoryBlock<int>(10);

        // Act
        var tensor = new Tensor<int>(memoryBlock, shape);

        // Assert
        tensor.Rank.Should().Be(2);
        tensor.ElementCount.Should().Be(10);
        tensor.IsScalar.Should().BeFalse();
        tensor.IsVector.Should().BeFalse();
        tensor.IsMatrix.Should().BeTrue();
        tensor.ReferenceCount.GetCount().Should().Be(1);

        tensor.Strides.Should().BeEquivalentTo(
            new int[]
            {
                5, 1
            });

        tensor.Dimensions.Should().BeEquivalentTo(
            new int[]
            {
                2, 5
            });

        tensor.Data.ToArray().Should().AllBeEquivalentTo(0);
        tensor.Data.Should().BeSameAs(memoryBlock);
    }

    [Fact]
    public void Ctor_GivenScalarShape_ShouldBeScalar()
    {
        // Arrange
        var shape = new Shape(1);

        // Act
        var tensor = new Tensor<int>(shape);

        // Assert
        tensor.Rank.Should().Be(1);
        tensor.ElementCount.Should().Be(1);
        tensor.IsScalar.Should().BeTrue();
        tensor.IsVector.Should().BeFalse();
        tensor.IsMatrix.Should().BeFalse();
        tensor.ReferenceCount.GetCount().Should().Be(1);

        tensor.Strides.Should().BeEquivalentTo(
            new int[]
            {
                1
            });

        tensor.Dimensions.Should().BeEquivalentTo(
            new int[]
            {
                1
            });

        tensor.Data.ToArray().Should().AllBeEquivalentTo(0);
    }

    [Fact]
    public void Ctor_GivenVectorShape_ShouldBeVector()
    {
        // Arrange
        var shape = new Shape(5);

        // Act
        var tensor = new Tensor<int>(shape);

        // Assert
        tensor.Rank.Should().Be(1);
        tensor.ElementCount.Should().Be(5);
        tensor.IsScalar.Should().BeFalse();
        tensor.IsVector.Should().BeTrue();
        tensor.IsMatrix.Should().BeFalse();
        tensor.ReferenceCount.GetCount().Should().Be(1);

        tensor.Strides.Should().BeEquivalentTo(
            new int[]
            {
                1
            });

        tensor.Dimensions.Should().BeEquivalentTo(
            new int[]
            {
                5
            });

        tensor.Data.ToArray().Should().AllBeEquivalentTo(0);
    }

    [Fact]
    public void Ctor_GivenMatrixShape_ShouldBeMatrix()
    {
        // Arrange
        var shape = new Shape(2, 3);

        // Act
        var tensor = new Tensor<int>(shape);

        // Assert
        tensor.Rank.Should().Be(2);
        tensor.ElementCount.Should().Be(6);
        tensor.IsScalar.Should().BeFalse();
        tensor.IsVector.Should().BeFalse();
        tensor.IsMatrix.Should().BeTrue();
        tensor.ReferenceCount.GetCount().Should().Be(1);

        tensor.Strides.Should().BeEquivalentTo(
            new int[]
            {
                3, 1
            });

        tensor.Dimensions.Should().BeEquivalentTo(
            new int[]
            {
                2, 3
            });

        tensor.Data.ToArray().Should().AllBeEquivalentTo(0);
    }
}