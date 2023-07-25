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
public class DivideShould
{
    private readonly ITensorBackend _sut;

    public DivideShould()
    {
        _sut = new ManagedTensorBackend();
    }

    [Fact]
    public void CorrectlyFillScalar_GivenScalarAndScalar()
    {
        // Arrange
        var left = new Scalar<int>(4, _sut);
        var right = new Scalar<int>(2, _sut);
        var result = new Scalar<int>(_sut);
        const int expectedResult = 2;

        // Act
        _sut.Arithmetic.Divide(left, right, result);

        // Assert
        result.Value.Should().Be(expectedResult);
    }

    [Fact]
    public void CorrectlyFillVector_GivenScalarAndVector()
    {
        // Arrange
        var left = new Scalar<int>(12);
        var right = Tensor.FromArray<int>(new int[] { 1, 2, 3 }).ToVector();
        var result = new Vector<int>(3);
        var expectedResult = new int[] { 12, 6, 4 };

        // Act
        _sut.Arithmetic.Divide(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillMatrix_GivenScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<int>(30);
        var right = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 10, 15, 30 } }).ToMatrix();
        var result = new Matrix<int>(2, 3);
        var expectedResult = new int[,] { { 30, 15, 10 }, { 3, 2, 1 } };

        // Act
        _sut.Arithmetic.Divide(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void CorrectlyFillTensor_GivenScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<int>(27720);

        var right = Tensor.FromArray<int>(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } })
            .ToTensor();
        var result = new Tensor<int>(new Shape(2, 2, 3));

        var expectedResult = new int[,,]
        {
            { { 27720, 13860, 9240 }, { 6930, 5544, 4620 } },
            { { 3960, 3465, 3080 }, { 2772, 2520, 2310 } }
        };

        // Act
        _sut.Arithmetic.Divide(left, right, result);

        // Assert
        result.ToArray().Should().BeEquivalentTo(expectedResult);
    }
}