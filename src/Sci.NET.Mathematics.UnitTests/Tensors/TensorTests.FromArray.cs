// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Tensors;

[SuppressMessage("Performance", "CA1814:Prefer jagged arrays over multidimensional", Justification = "Test data")]
public partial class TensorTests
{
    [Fact]
    public void FromArray_GivenShapeAndValues_ReturnsExpectedResult()
    {
        // Arrange
        var shape = new Shape(2, 3);

        var values = new int[]
        {
            1, 2, 3, 4, 5, 6
        };

        // Act
        var tensor = Tensor.FromArray(shape, values);

        // Assert
        tensor.Rank.Should().Be(2);
        tensor.ElementCount.Should().Be(6);
        tensor.IsScalar.Should().BeFalse();
        tensor.IsVector.Should().BeFalse();
        tensor.IsMatrix.Should().BeTrue();

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

        tensor.Data.ToArray().Should().BeEquivalentTo(values);
    }

    [Fact]
    public void FromArray_GivenShapeAndInvalidValues_ThrowsException()
    {
        // Arrange
        var shape = new Shape(2, 3);

        var values = new int[]
        {
            1, 2, 3
        };

        // Act
        var act = () => Tensor.FromArray(shape, values);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void FromArray_GivenArray_ReturnsExpectedResult()
    {
        // Arrange
        var values = new int[,]
        {
            {
                1, 2, 3
            },
            {
                4, 5, 6
            }
        };

        // Act
        var tensor = Tensor.FromArray<int>(values);

        // Assert
        tensor.Rank.Should().Be(2);
        tensor.ElementCount.Should().Be(6);
        tensor.IsScalar.Should().BeFalse();
        tensor.IsVector.Should().BeFalse();
        tensor.IsMatrix.Should().BeTrue();

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

        tensor.Data.ToArray().Should().BeEquivalentTo(
            new int[]
            {
                1, 2, 3, 4, 5, 6
            });
    }

    [Fact]
    public void FromArray_GivenArrayWithInvalidType_ThrowsException()
    {
        // Arrange
        var values = new int[,]
        {
            {
                1, 2, 3
            },
            {
                4, 5, 6
            }
        };

        // Act
        var act = () => Tensor.FromArray<double>(values);

        // Assert
        act.Should().Throw<ArgumentException>();
    }
}