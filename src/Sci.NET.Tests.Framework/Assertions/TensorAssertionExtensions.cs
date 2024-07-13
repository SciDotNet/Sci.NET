// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Tests.Framework.Assertions;

/// <summary>
/// Extension methods for <see cref="ITensor{TNumber}" />.
/// </summary>
public static class TensorAssertionExtensions
{
    /// <summary>
    /// Extension method to create a <see cref="TensorAssertions{TNumber}" /> object for the given <see cref="ITensor{TNumber}" />.
    /// </summary>
    /// <param name="tensor">The tensor to create assertions for.</param>
    /// <typeparam name="TNumber">The type of the tensor.</typeparam>
    /// <returns>A <see cref="TensorAssertions{TNumber}" /> object for the given <see cref="ITensor{TNumber}" />.</returns>
    public static TensorAssertions<TNumber> Should<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return new(tensor);
    }
}