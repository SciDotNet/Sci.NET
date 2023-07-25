// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Backends.Managed.Arithmetic;

[SuppressMessage("Performance", "CA1814:Prefer jagged arrays over multidimensional", Justification = "Test Class")]
[SuppressMessage(
    "StyleCop.CSharp.LayoutRules",
    "SA1500:Braces for multi-line statements should not share line",
    Justification = "Test Class")]
public class SubtractShould
{
    private readonly ITensorBackend _sut;

    public SubtractShould()
    {
        _sut = new ManagedTensorBackend();
    }

    [Fact]
    public void CorrectlyFillScalar_GivenScalarAndScalar()
    {
        // Arrange
        var left = new Scalar<int>(1, _sut);
        var right = new Scalar<int>(1, _sut);
        var result = new Scalar<int>(_sut);

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.Value.Should().Be(0);
    }

    [Fact]
    public void CorrectlyFillVector_GivenScalarAndVector()
    {
        // Arrange
        var left = new Scalar<int>(1);
        var right = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        var result = new Vector<int>(3);
        var expectedResult = new int[] { 0, -1, -2 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillMatrix_GivenScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<int>(1);
        var right = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        var result = new Matrix<int>(2, 3);
        var expectedResult = new int[] { 0, -1, -2, -3, -4, -5 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillsTensor_GivenScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<int>(1);

        var right = Tensor.FromArray<int>(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } })
            .ToTensor();

        var result = new Tensor<int>(new Shape(2, 2, 3));

        var expectedResult = new int[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillVector_GivenVectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        var right = new Scalar<int>(1);
        var result = new Vector<int>(3);
        var expectedResult = new int[] { 0, 1, 2 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillsMatrix_GivenVectorAndMatrix()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        var right = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        var result = new Matrix<int>(2, 3);
        var expectedResult = new int[] { 0, -1, -2, -3, -2, -4 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillMatrix_GivenMatrixAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        var right = new Scalar<int>(1);
        var result = new Matrix<int>(2, 3);
        var expectedResult = new int[] { 0, 1, 2, 3, 4, 5 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillVector_GivenVectorAndVector()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        var right = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        var result = new Vector<int>(3);
        var expectedResult = new int[] { 0, 0, 0 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillMatrix_GivenMatrixAndVector()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        var right = Tensor.FromArray<int>(new int[] { 1, 2 }).ToVector();
        var result = new Matrix<int>(2, 3);
        var expectedResult = new int[] { 0, 1, 2, 2, 3, 4 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillMatrix_GivenMatrixAndMatrix()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        var right = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        var result = new Matrix<int>(2, 3);
        var expectedResult = new int[] { 0, 0, 0, 0, 0, 0 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillsTensor_GivenTensorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } })
            .ToTensor();
        var right = new Scalar<int>(1);
        var result = new Tensor<int>(new Shape(2, 2, 3));
        var expectedResult = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillsTensor_GivenTensorAndTensor()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[,,] { { { 5, 6, 7 }, { 8, 9, 10 } }, { { 11, 12, 13 }, { 14, 15, 16 } } })
            .ToTensor();

        var right = Tensor.FromArray<int>(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } })
            .ToTensor();

        var result = new Tensor<int>(new Shape(2, 2, 3));
        var expectedResult = new int[] { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };

        // Act
        _sut.Arithmetic.Subtract(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }
}