// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130 // API accessibility
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extension methods to reshape a <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class ReshapeExtensions
{
    /// <summary>
    /// Reshapes a <see cref="ITensor{TNumber}"/> to a new shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to reshape.</param>
    /// <param name="shape">The new shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The reshaped <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Throws if the new <paramref name="shape"/> is incompatible with
    /// the <see cref="ITensor{TNumber}"/>.</exception>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Reshape<TNumber>(this ITensor<TNumber> tensor, Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReshapeService()
            .Reshape(tensor, shape);
    }

    /// <inheritdoc cref="Reshape{TNumber}(Sci.NET.Mathematics.Tensors.ITensor{TNumber},Sci.NET.Mathematics.Tensors.Shape)"/>>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Reshape<TNumber>(this ITensor<TNumber> tensor, params int[] shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return Reshape(tensor, new Shape(shape));
    }
}