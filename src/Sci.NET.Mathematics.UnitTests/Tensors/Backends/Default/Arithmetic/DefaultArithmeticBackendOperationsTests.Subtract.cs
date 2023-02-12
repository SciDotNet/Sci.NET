// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Backends.Default.Arithmetic;

public partial class DefaultArithmeticBackendOperationsTests
{
    [Fact]
    public void Subtract_GivenTwoTensorsOfSameShape_ReturnsExpectedResult()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3)_right.tensor");
        var expected = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3)_result.tensor");

        // Act
        var result = _sut.Subtract(left, right);

        // Assert
        result.Should().BeEqualTo(expected);
    }

    [Fact]
    public void Subtract_GivenTensorAndScalarTensor_ReturnsExpectedResult()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3-4)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(1)_right.tensor");
        var expected = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3-4)_(1)_result.tensor");

        // Act
        var result = _sut.Subtract(left, right);

        // Assert
        result.Should().BeEqualTo(expected);
    }

    [Fact]
    public void Subtract_GivenScalarTensorAndTensor_ReturnsExpectedResult()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(1)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3-4)_right.tensor");
        var expected = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(1)_(2-3-4)_result.tensor");

        // Act
        var result = _sut.Subtract(left, right);

        // Assert
        result.Should().BeEqualTo(expected);
    }

    [Fact]
    public void Subtract_GivenTwoTensorsOfDifferentShape_Throws()
    {
        // Arrange
        var left = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3)_left.tensor");
        var right = Tensor.Load<float>(@".\Data\Tensors\Operations\subtract_float_(2-3-4)_right.tensor");

        // Act
        var action = () => _sut.Subtract(left, right);

        // Assert
        action.Should().Throw<InvalidShapeException>();
    }
}