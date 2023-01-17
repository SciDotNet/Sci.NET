// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.Unmanaged;
using Sci.NET.Mathematics.BLAS;
using Sci.NET.Mathematics.Tensors.Backends.Default;

namespace Sci.NET.Mathematics.Tensors.Backends;

/// <summary>
/// Provides an interface for tensor backends.
/// </summary>
[PublicAPI]
public abstract class TensorBackend
{
    private static TensorBackend? _instance;

    /// <summary>
    /// Gets the default backend implementation.
    /// </summary>
    public static TensorBackend Instance => _instance ??= new DefaultTensorBackend();

    /// <summary>
    /// Gets the memory manager used by the backend.
    /// </summary>
    public abstract INativeMemoryManager MemoryManager { get; }

    /// <summary>
    /// Gets the BLAS implementation used by the backend.
    /// </summary>
    public abstract IBlasProvider BlasProvider { get; }

    /// <summary>
    /// Sets the backend implementation.
    /// </summary>
    /// <typeparam name="TBackend">The type of backend.</typeparam>
    /// <exception cref="InvalidOperationException">Throws when the backend cannot be changed.</exception>
    public static void Use<TBackend>()
        where TBackend : TensorBackend, new()
    {
        if (_instance is not null)
        {
            throw new InvalidOperationException("The backend cannot be changed once it has been set.");
        }

        _instance = new TBackend();
    }

    /// <summary>
    /// Creates a new tensor with the specified tensorShape.
    /// </summary>
    /// <param name="tensorShape">The tensorShape of the tensor.</param>
    /// <typeparam name="TTensor">The number type of the tensor.</typeparam>
    /// <returns>A reference to a new tensor object with the given tensorShape.</returns>
    public virtual TypedMemoryHandle<TTensor> Create<TTensor>(Shape tensorShape)
        where TTensor : unmanaged, INumber<TTensor>
    {
        return MemoryManager.Allocate<TTensor>(tensorShape.ElementCount);
    }

    /// <summary>
    /// Frees the memory allocated by the backend.
    /// </summary>
    /// <param name="handle">The handle to the tensor to free.</param>
    /// <typeparam name="TTensor">The number type of the tensor.</typeparam>
    public virtual void Free<TTensor>(TypedMemoryHandle<TTensor> handle)
        where TTensor : unmanaged
    {
        MemoryManager.Free(handle);
    }

    /// <summary>
    /// Copies the data from the source tensor to the destination tensor.
    /// </summary>
    /// <param name="source">The source tensor.</param>
    /// <param name="destination">The destination tensor.</param>
    /// <typeparam name="TTensor">The number type of the tensor.</typeparam>
    /// <exception cref="ArgumentException">The source and destination tensors are incompatible.</exception>
    public virtual void Copy<TTensor>(Tensor<TTensor> source, Tensor<TTensor> destination)
        where TTensor : unmanaged, INumber<TTensor>
    {
        if (!source.Dimensions.SequenceEqual(destination.Dimensions))
        {
            throw new ArgumentException("The source and destination tensors must have the same tensorShape.");
        }

        MemoryManager.Copy(source.Handle, destination.Handle, source.ElementCount);
    }

    /// <summary>
    /// Performs a matrix multiplication of the two tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the matrix multiplication between the two operands.</returns>
    public virtual ITensor<TNumber> MatrixMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(new Shape(left.Dimensions[0], right.Dimensions[1]));

        BlasProvider.Gemm(
            left.TransposeType,
            right.TransposeType,
            left.Dimensions[0],
            right.Dimensions[1],
            left.Dimensions[1],
            TNumber.One,
            left.Handle,
            left.Dimensions[0],
            right.Handle,
            right.Dimensions[0],
            TNumber.Zero,
            result.Handle,
            left.Dimensions[0]);

#pragma warning disable RCS1124
        var res = result.Handle.CopyToArray(result.ElementCount);
#pragma warning restore RCS1124

        _ = res;

        return result;
    }

    /// <summary>
    /// For vectors, the inner product of two <see cref="ITensor{TNumber}"/>s is calculated,
    /// for higher dimensions then the sum product over the last axes are calculated.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the inner product operation.</returns>
    /// <exception cref="ArgumentException">Throws when the operand shapes are incompatible with the
    /// inner product operation.</exception>
    public abstract ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of the left and right operands.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the operation.</returns>
    public abstract ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;
}