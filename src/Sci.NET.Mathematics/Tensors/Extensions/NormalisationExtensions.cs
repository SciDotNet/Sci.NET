// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
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
    [DebuggerStepThrough]
    public static Matrix<TNumber> BatchNorm1dForward<TNumber>(this Matrix<TNumber> input, Vector<TNumber> scale, Vector<TNumber> bias)
        where TNumber : unmanaged, INumber<TNumber>, IRootFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetNormalisationService()
            .BatchNorm1dForward(input, scale, bias);
    }

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
            .GetNormalisationService()
            .Clip(tensor, min, max);
    }

    /// <inheritdoc cref="Clip{TNumber}(Sci.NET.Mathematics.Tensors.ITensor{TNumber},TNumber,TNumber)"/>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Clip<TNumber>(this Scalar<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetNormalisationService()
            .Clip(value, min, max)
            .ToScalar();
    }

    /// <inheritdoc cref="Clip{TNumber}(Sci.NET.Mathematics.Tensors.ITensor{TNumber},TNumber,TNumber)"/>
    [DebuggerStepThrough]
    public static Vector<TNumber> Clip<TNumber>(this Vector<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetNormalisationService()
            .Clip(value, min, max)
            .ToVector();
    }

    /// <inheritdoc cref="Clip{TNumber}(Sci.NET.Mathematics.Tensors.ITensor{TNumber},TNumber,TNumber)"/>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Clip<TNumber>(this Matrix<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetNormalisationService()
            .Clip(value, min, max)
            .ToMatrix();
    }

    /// <inheritdoc cref="Clip{TNumber}(Sci.NET.Mathematics.Tensors.ITensor{TNumber},TNumber,TNumber)"/>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Clip<TNumber>(this Tensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetNormalisationService()
            .Clip(value, min, max)
            .ToTensor();
    }
}