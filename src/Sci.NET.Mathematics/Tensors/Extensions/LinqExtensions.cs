// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// An interface providing LINQ methods for <see cref="ITensor{TNumber}"/> instances.
/// </summary>
[PublicAPI]
public static class LinqExtensions
{
    /// <summary>
    /// Performs a pointwise mapping operation on a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> instance to map.</param>
    /// <param name="predicate">The function to perform on each element.</param>
    /// <typeparam name="TTensor">The concrete <see cref="ITensor{TNumber}"/> type.</typeparam>
    /// <typeparam name="TNumber">The concrete <typeparamref name="TNumber"/> type.</typeparam>
    /// <returns>The mapped <typeparamref name="TTensor"/>.</returns>
    [DebuggerStepThrough]
    public static TTensor Map<TTensor, TNumber>(this TTensor tensor, Func<TNumber, TNumber> predicate)
        where TTensor : class, ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetLinqService()
            .Map(tensor, predicate);
    }
}