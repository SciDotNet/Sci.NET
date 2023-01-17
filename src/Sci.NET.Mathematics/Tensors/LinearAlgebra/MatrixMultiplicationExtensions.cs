// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra;

/// <summary>
/// Extension methods for matrix multiplication.
/// </summary>
[PublicAPI]
public static class MatrixMultiplicationExtensions
{
    /// <inheritdoc cref="TensorBackend.MatrixMultiply{TNumber}"/>
    public static ITensor<TNumber> MatrixMultiply<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorBackend.Instance.MatrixMultiply(left, right);
    }
}