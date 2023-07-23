// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130 // API accessibility
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extension methods to permute a <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class PermutationExtensions
{
    /// <summary>
    /// Permutes the <see cref="ITensor{TNumber}"/>, rearranging the indices
    /// in the order passed as a parameter.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to permute.</param>
    /// <param name="permutation">The new order of the indices.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The permuted <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Throws when the <paramref name="permutation"/>
    /// indices are invalid.</exception>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Permute<TNumber>(this ITensor<TNumber> tensor, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPermutationService()
            .Permute(tensor, permutation);
    }
}