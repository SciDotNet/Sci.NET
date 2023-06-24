// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Pointwise;

/// <summary>
/// A service for performing pointwise operations on tensors.
/// </summary>
[PublicAPI]
public interface ILinqService
{
    /// <summary>
    /// Performs a pointwise mapping operation on a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> instance to map.</param>
    /// <param name="func">The function to perform on each element.</param>
    /// <typeparam name="TTensor">The concrete <see cref="ITensor{TNumber}"/> type.</typeparam>
    /// <typeparam name="TNumber">The concrete <typeparamref name="TNumber"/> type.</typeparam>
    /// <returns>The mapped <typeparamref name="TTensor"/>.</returns>
    public TTensor Map<TTensor, TNumber>(TTensor tensor, Func<TNumber, TNumber> func)
        where TTensor : class, ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>;
}