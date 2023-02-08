// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// Extension methods for casting an <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class CastingExtensions
{
    /// <summary>
    /// Casts the <see cref="ITensor{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The input type.</typeparam>
    /// <typeparam name="TOut">The output type.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> with values cast to the specified type.</returns>
    public static ITensor<TOut> Cast<TIn, TOut>(this ITensor<TIn> tensor)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var result = new Tensor<TOut>(tensor.GetShape());
        var length = tensor.Data.Length;
        var data = tensor.Data;
        var resultData = result.Data;

        LazyParallelExecutor.For(
            0,
            length,
            1000,
            i => resultData[i] = TOut.CreateChecked(data[i]));

        return result;
    }

    /// <summary>
    /// Casts the <see cref="ITensor{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to convert.</param>
    /// <typeparam name="TNumber">The type to convert to.</typeparam>
    /// <returns>The <see cref="ITensor{TNumber}"/> cast to the specified type.</returns>
    public static ITensor<TNumber> Cast<TNumber>(this ITensor<byte> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.GetShape());
        var length = tensor.Data.Length;
        var data = tensor.Data;
        var resultData = result.Data;

        LazyParallelExecutor.For(
            0,
            length,
            1000,
            i => resultData[i] = TNumber.CreateChecked(data[i]));

        return result;
    }
}