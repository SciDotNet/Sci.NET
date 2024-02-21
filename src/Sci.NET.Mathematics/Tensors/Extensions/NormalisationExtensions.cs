// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extension methods for normalisation operations.
/// </summary>
[PublicAPI]
public static class NormalisationExtensions
{
    /// <summary>
    /// Normalises the input tensor using batch normalisation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="scale">The weight tensor.</param>
    /// <param name="bias">The bias tensor.</param>
    /// <typeparam name="TNumber">The type of the tensor.</typeparam>
    /// <returns>The normalised tensor.</returns>
    public static Matrix<TNumber> BatchNorm1dForward<TNumber>(this Matrix<TNumber> input, Vector<TNumber> scale, Vector<TNumber> bias)
        where TNumber : unmanaged, INumber<TNumber>, IRootFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetNormalisationService()
            .BatchNorm1dForward(input, scale, bias);
    }
}