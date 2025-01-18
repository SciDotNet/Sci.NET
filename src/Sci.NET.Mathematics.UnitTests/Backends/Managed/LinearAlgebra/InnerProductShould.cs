// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Backends.Managed.LinearAlgebra;

public class InnerProductShould
{
    private readonly ManagedTensorBackend _sut;

    public InnerProductShould()
    {
        _sut = new ManagedTensorBackend();
    }

    [Fact]
    public void ReturnCorrectResult()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, backend: _sut).ToVector();
        var right = Tensor.FromArray<int>(new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, backend: _sut).ToVector();
        var result = new Scalar<int>();
        const int expected = 220;

        // Act
        _sut.LinearAlgebra.InnerProduct(left, right, result);

        // Assert
        result.Memory.ToSystemMemory()[0].Should().Be(expected);
    }
}