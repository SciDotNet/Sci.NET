// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Managed;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Provides static methods for manipulating tensors.
/// </summary>
[PublicAPI]
public static class Tensor
{
    /// <summary>
    /// Gets the default backend for tensors.
    /// </summary>
    public static ITensorBackend DefaultBackend => new ManagedTensorBackend();

    /// <summary>
    /// Creates a tensor with the specified dimensions and values.
    /// </summary>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="array">The array of values for the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="backend">The backend instance for the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given values and shape.</returns>
    /// <exception cref="ArgumentException">Throws when the array does not contain the same number of elements as the shape.</exception>
    public static ITensor<TNumber> FromArray<TNumber>(Shape shape, TNumber[] array, ITensorBackend? backend = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        backend ??= DefaultBackend;

        if (array.LongLength != shape.ElementCount)
        {
            throw new ArgumentException("The array length must match the shape size.", nameof(array));
        }

        var handle = backend.Storage.Allocate<TNumber>(shape);
        handle.CopyFrom(array);

        return new Tensor<TNumber>(handle, shape, backend);
    }

    /// <summary>
    /// Creates a tensor with the specified dimensions and values.
    /// </summary>
    /// <param name="array">The values to assign to the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="backend">The <see cref="ITensorBackend"/> to use.</param>
    /// <typeparam name="TNumber">The type of element in the <see cref="Array"/>.</typeparam>
    /// <returns>The <see cref="Array"/> as a <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">The <see cref="Array"/> elements were not
    /// the same as <typeparamref name="TNumber"/>.</exception>
    public static ITensor<TNumber> FromArray<TNumber>(
        Array array,
        ITensorBackend? backend = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        backend ??= DefaultBackend;
        var dims = new int[array.Rank];

        for (var i = 0; i < array.Rank; i++)
        {
            dims[i] = array.GetLength(i);
        }

        var shape = new Shape(dims);
        var flattened = new TNumber[shape.ElementCount];

        for (var i = 0; i < shape.ElementCount; i++)
        {
            var value = array.GetValue(shape.GetIndicesFromLinearIndex(i));

            flattened[i] = value is TNumber number
                ? number
                : throw new ArgumentException(
                    "The array elements must be of the same type as the tensor.",
                    nameof(array));
        }

        return FromArray(shape, flattened, backend);
    }

    /// <summary>
    /// Slices a <see cref="ITensor{TNumber}"/> at the specified <paramref name="indices"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to slice.</param>
    /// <param name="indices">The indices to slice at.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The specified slice.</returns>
    public static ITensor<TNumber> Slice<TNumber>(ITensor<TNumber> tensor, params int[] indices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var newShape = tensor.Shape.Slice(indices);
        return new Tensor<TNumber>(tensor, newShape);
    }
}