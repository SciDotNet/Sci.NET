// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Backends.Managed.LinearAlgebra;

[SuppressMessage(
    "Performance",
    "CA1814:Prefer jagged arrays over multidimensional",
    Justification = "This is a test")]
[SuppressMessage(
    "StyleCop.CSharp.LayoutRules",
    "SA1500:Braces for multi-line statements should not share line",
    Justification = "This is a test")]
public class MatrixMultiplyShould
{
    private readonly ManagedTensorBackend _sut;

    public MatrixMultiplyShould()
    {
        _sut = new ManagedTensorBackend();
    }

    [Fact]
    public void ReturnExpectedResults_GivenValidMatrices()
    {
        // Arrange
        var a = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, backend: _sut).ToMatrix();
        var b = Tensor.FromArray<int>(new int[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } }, backend: _sut).ToMatrix();
        var result = new Matrix<int>(a.Rows, b.Columns);
        var expected = new int[] { 58, 64, 139, 154 };

        // Act
        _sut.LinearAlgebra.MatrixMultiply(a, b, result);
        var resultArray = result.Memory.ToArray();

        // Assert
        resultArray.Should().BeEquivalentTo(expected);
    }
}