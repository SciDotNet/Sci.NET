// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors.Exceptions;
using Sci.NET.Mathematics.Tensors.Random;

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
    public static ITensorBackend DefaultBackend { get; private set; } = new ManagedTensorBackend();

    /// <summary>
    /// Gets the default random service for tensors.
    /// </summary>
    public static IRandomService Random { get; } = TensorServiceProvider.GetTensorOperationServiceProvider().GetRandomService();

    /// <summary>
    /// Sets the default backend for tensors.
    /// </summary>
    /// <typeparam name="TBackend">The type of backend to use.</typeparam>
    public static void SetDefaultBackend<TBackend>()
        where TBackend : ITensorBackend, new()
    {
        DefaultBackend = new TBackend();
    }

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
    /// <exception cref="ArgumentException">Throws when invalid indices are specified.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Throws when the indices are out of range.</exception>
    public static ITensor<TNumber> Slice<TNumber>(ITensor<TNumber> tensor, params int[] indices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var shape = tensor.Shape.Slice(indices);
        var result = new Tensor<TNumber>(new Shape(shape.Dimensions), tensor.Backend);

        result.Memory.BlockCopyFrom(
            tensor.Memory,
            shape.DataOffset,
            0,
            shape.ElementCount);

        return result;
    }

    /// <summary>
    /// Clones a <see cref="ITensor{TNumber}"/> with the same values and shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to clone.</param>
    /// <typeparam name="TTensor">The runtime type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>An empty <typeparamref name="TTensor"/> with empty values.</returns>
    /// <exception cref="InvalidOperationException">Throws when the <see cref="ITensor{TNumber}"/> could not be cloned.</exception>
    public static TTensor CloneEmpty<TTensor, TNumber>(TTensor tensor)
        where TTensor : class, ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
#pragma warning restore CA2000
        return result as TTensor ?? throw new InvalidOperationException();
    }

    /// <summary>
    /// Loads a tensor from the specified path.
    /// </summary>
    /// <param name="path">The path to load the tensor from.</param>
    /// <typeparam name="TNumber">The number type of the tensor.</typeparam>
    /// <returns>The loaded tensor.</returns>
    public static ITensor<TNumber> Load<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .Load<TNumber>(path);
    }

    /// <summary>
    /// Loads a tensor from the specified buffer.
    /// </summary>
    /// <param name="stream">The buffer to load the tensor from.</param>
    /// <typeparam name="TNumber">The number type of the tensor.</typeparam>
    /// <returns>The loaded tensor.</returns>
    public static ITensor<TNumber> LoadFromBuffer<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .Load<TNumber>(stream);
    }

    /// <summary>
    /// Creates a <see cref="ITensor{TNumber}"/> with the specified dimensions which is filled with zeros..
    /// </summary>
    /// <param name="dimensions">The dimensions of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given dimensions and filled with zeros.</returns>
    public static ITensor<TNumber> Zeros<TNumber>(params int[] dimensions)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return new Tensor<TNumber>(new Shape(dimensions));
    }

    /// <summary>
    /// Creates a <see cref="ITensor{TNumber}"/> with the specified <paramref name="shape"/> which is filled with zeros..
    /// </summary>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given <paramref name="shape"/> and filled with zeros.</returns>
    public static ITensor<TNumber> Zeros<TNumber>(Shape shape, IDevice? device = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        device ??= new CpuComputeDevice();
        return new Tensor<TNumber>(shape, device.GetTensorBackend());
    }

    /// <summary>
    /// Creates a <see cref="ITensor{TNumber}"/> with the specified dimensions which is filled with ones.
    /// </summary>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given <paramref name="shape"/> and filled with ones.</returns>
    public static ITensor<TNumber> Ones<TNumber>(Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(shape);

        result.Memory.Fill(TNumber.One);

        return result;
    }

    /// <summary>
    /// Creates a <see cref="ITensor{TNumber}"/> with the specified dimensions which is filled with ones.
    /// </summary>
    /// <param name="dimensions">The dimensions of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given dimensions and filled with ones.</returns>
    public static ITensor<TNumber> Ones<TNumber>(params int[] dimensions)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return Ones<TNumber>(new Shape(dimensions));
    }

    /// <summary>
    /// Creates a <see cref="ITensor{TNumber}"/> with the specified dimensions which is filled with the specified value.
    /// </summary>
    /// <param name="value">The value to fill the <see cref="ITensor{TNumber}"/> with.</param>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given <paramref name="shape"/> and filled with the specified value.</returns>
    public static ITensor<TNumber> FillWith<TNumber>(TNumber value, Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(shape);

        result.Memory.Fill(value);

        return result;
    }

    /// <summary>
    /// Creates a <see cref="ITensor{TNumber}"/> with the specified dimensions which is filled with the specified value.
    /// </summary>
    /// <param name="value">The value to fill the <see cref="ITensor{TNumber}"/> with.</param>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the given <paramref name="shape"/> and filled with the specified value.</returns>
    public static ITensor<TNumber> FillWith<TNumber>(TNumber value, params int[] shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return FillWith(value, new Shape(shape));
    }

    /// <summary>
    /// Overwrites the values of the <paramref name="tensor"/> with the values of the <paramref name="other"/> tensor.
    /// This should only be used on rare occasions where referential integrity is required (I.E weights in a neural network).
    /// </summary>
    /// <param name="tensor">The tensor to overwrite.</param>
    /// <param name="other">The tensor to overwrite with.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    /// <returns>The overwritten tensor.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shapes of the tensors are not equal.</exception>
    public static ITensor<TNumber> InPlaceOverwrite<TNumber>(this ITensor<TNumber> tensor, ITensor<TNumber> other)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (tensor.Shape != other.Shape)
        {
            throw new InvalidShapeException($"The shapes of the tensors must be equal but were {tensor.Shape} and {other.Shape}.");
        }

        other.Memory.CopyTo(tensor.Memory);
        return tensor;
    }

    /// <summary>
    /// Converts a <see cref="ITensor{TNumber}"/> to an array.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to convert.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <see cref="ITensor{TNumber}"/> as an array.</returns>
    /// <exception cref="InvalidOperationException">Throws when the <see cref="ITensor{TNumber}"/> is a <see cref="Scalar{TNumber}"/>.</exception>
    public static unsafe Array ToArray<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (tensor.Shape.IsScalar)
        {
            throw new InvalidOperationException("Cannot convert a scalar to an array.");
        }

        var result = Array.CreateInstance(typeof(TNumber), tensor.Shape.Dimensions);
        result.Initialize();

        var startIndex = tensor.Shape.DataOffset;
        var endIndex = startIndex + tensor.Shape.ElementCount;
        var bytesToCopy = Unsafe.SizeOf<TNumber>() * (endIndex - startIndex);
        var systemMemoryClone = tensor.Memory.ToSystemMemory();

        var sourcePointer = Unsafe.AsPointer(ref Unsafe.Add(ref Unsafe.AsRef<TNumber>(systemMemoryClone.ToPointer()), (nuint)startIndex));
        var destinationPointer = Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(result));

        Buffer.MemoryCopy(
            sourcePointer,
            destinationPointer,
            Unsafe.SizeOf<TNumber>() * result.LongLength,
            bytesToCopy);

        return result;
    }

    /// <summary>
    /// Determines whether the <paramref name="tensor"/> is effectively a scalar, meaning that it is a scalar or has all dimensions equal to 1.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to check.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>Whether the <paramref name="tensor"/> is effectively a <see cref="Scalar{TNumber}"/>.</returns>
    public static bool IsEffectiveScalar<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return tensor.Shape.IsScalar ||
               tensor.Shape.Dimensions.All(x => x == 1);
    }

    /// <summary>
    /// Determines whether the <paramref name="tensor"/> is effectively a vector, meaning that it is a vector or has exactly one dimension that is not equal to 1.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to check.</param>`
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>Whether the <paramref name="tensor"/> is effectively a <see cref="Vector{TNumber}"/>.</returns>
    public static bool IsEffectiveVector<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return tensor.Shape.IsVector ||
               tensor.Shape.Dimensions.Count(x => x != 1) == 1;
    }

    /// <summary>
    /// Determines whether the <paramref name="tensor"/> is effectively a matrix, meaning that it is a matrix or has exactly two dimensions that are not equal to 1.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to check.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>Whether the <paramref name="tensor"/> is effectively a <see cref="Matrix{TNumber}"/>.</returns>
    public static bool IsEffectiveMatrix<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return tensor.Shape.IsMatrix ||
               tensor.Shape.Dimensions.Count(x => x != 0) == 2;
    }
}