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
    /// Clips the values of the <see cref="ITensor{TNumber}"/> to the specified range.
    /// </summary>
    /// <param name="tensor">The the <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <param name="min">The minimum value to clip to.</param>
    /// <param name="max">The maximum value to clip to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The clipped <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Clip<TNumber>(this ITensor<TNumber> tensor, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetLinqService()
            .Clip(tensor, min, max);
    }
}