// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Slicing;

public class SliceShould
{
    [Fact]
    public void ThrowException_GivenScalar()
    {
        // Arrange
        var scalar = new Scalar<float>(1.0f);

        // Act
        var act = () => scalar[0];

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ReturnCorrectValue_GivenVector()
    {
        // Arrange
        var vector = Tensor.FromArray<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }).ToVector();

        // Act
        var slice0 = vector[0].Value;
        var slice1 = vector[1].Value;
        var slice2 = vector[2].Value;
        var slice3 = vector[3].Value;

        // Assert
        slice0.Should().Be(1.0f);
        slice1.Should().Be(2.0f);
        slice2.Should().Be(3.0f);
        slice3.Should().Be(4.0f);
    }

    [Fact]
    public void ReturnCorrectValue_GivenMatrix()
    {
        // Arrange
        var matrix = Tensor.FromArray<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }).Reshape(2, 3).ToMatrix();
        var slice0Expected = Tensor.FromArray<float>(new float[] { 1.0f, 2.0f, 3.0f }).ToVector().ToArray();
        var slice1Expected = Tensor.FromArray<float>(new float[] { 4.0f, 5.0f, 6.0f }).ToVector().ToArray();

        // Act
        var slice0 = matrix[0];
        var slice1 = matrix[1];

        // Assert
        slice0.ToArray().Should().BeEquivalentTo(slice0Expected);
        slice1.ToArray().Should().BeEquivalentTo(slice1Expected);
    }
}