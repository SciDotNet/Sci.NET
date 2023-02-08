// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Tensors;

public partial class TensorTests
{
    [Fact]
    public void Zeros_GivenShape_ReturnsExpectedResult()
    {
        // Arrange
        var shape = new Shape(2, 3);

        // Act
        var tensor = Tensor.Zeros<int>(shape);

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
                0, 0, 0, 0, 0, 0
            });
    }
}