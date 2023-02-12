// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Backends.Default.Arithmetic;

public partial class DefaultArithmeticBackendOperationsTests
{
    [Fact]
    public void Add_GivenTwoTensorsOfSameShape_ReturnsExpectedResult()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3-4)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3-4)_right.tensor");
        var expected = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3-4)_result.tensor");

        // Act
        var result = _sut.Add(left, right);

        // Assert
        result.Should().BeEqualTo(expected);
    }

    [Fact]
    public void Add_GivenTwoTensorsOfDifferentShape_Throws()
    {
        // Arrange
        var left = Tensor.Random.Uniform(new Shape(2, 3, 4), 0.0f, 1.0f, 0);
        var right = Tensor.Random.Uniform(new Shape(3, 4), 0.0f, 1.0f, 0);

        // Act
        var action = () => _sut.Add(left, right);

        // Assert
        action.Should().Throw<InvalidShapeException>();
    }

    [Fact]
    public void Add_GivenTensorAndScalarTensor_ReturnsExpectedResult()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float(1)_right.tensor");
        var expected = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3)_(1)_result.tensor");

        // Act
        var result = _sut.Add(left, right);

        // Assert
        result.Should().BeEqualTo(expected);
    }

    [Fact]
    public void Add_GivenTensorScalarAndTensor_ReturnsExpectedResult()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float(1)_right.tensor");
        var expected = Tensor.Load<float>(@".\Data\Tensors\Operations\add_float_(2-3)_(1)_result.tensor");

        // Act
        var result = _sut.Add(right, left);

        // Assert
        result.Should().BeEqualTo(expected);
    }
}