// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
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
    /// <typeparam name="TNumber">The number type of the tensor.</typeparam>
    /// <returns>A reference to a new tensor object with the given tensorShape.</returns>
    public abstract IMemoryBlock<TNumber> Create<TNumber>(Shape tensorShape)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Frees the memory allocated by the backend.
    /// </summary>
    /// <param name="handle">The handle to the tensor to free.</param>
    /// <typeparam name="TNumber">The number type of the tensor.</typeparam>
    public abstract void Free<TNumber>(IMemoryBlock<TNumber> handle)
        where TNumber : unmanaged;

    /// <summary>
    /// Performs a matrix multiplication of the two tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the matrix multiplication between the two operands.</returns>
    public abstract ITensor<TNumber> MatrixMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

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

    /// <summary>
    /// Elementwise sine of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the sin operation.</returns>
    public abstract ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Elementwise cosine of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the cosine operation.</returns>
    public abstract ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Elementwise tangent of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the tangent operation.</returns>
    public abstract ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;
}