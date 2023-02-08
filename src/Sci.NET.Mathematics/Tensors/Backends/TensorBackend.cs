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
    /// Gets the current backend implementation.
    /// </summary>
    public static TensorBackend Instance => _instance ??= new DefaultTensorBackend();

    /// <summary>
    /// Gets the random backend implementation.
    /// </summary>
    public abstract IRandomBackendOperations Random { get; }

    /// <summary>
    /// Gets the linear algebra backend implementation.
    /// </summary>
    public abstract ILinearAlgebraBackendOperations LinearAlgebra { get; }

    /// <summary>
    /// Gets the trigonometry backend implementation.
    /// </summary>
    public abstract ITrigonometryBackendOperations Trigonometry { get; }

    /// <summary>
    /// Gets the arithmetic backend implementation.
    /// </summary>
    public abstract IArithmeticBackendOperations Arithmetic { get; }

    /// <summary>
    /// Gets the mathematical backend implementation.
    /// </summary>
    public abstract IMathematicalBackendOperations MathematicalOperations { get; }

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
    /// Performs a scalar multiplication of the left and right operands.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the operation.</returns>
    public abstract ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of the left and right operands.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the operation.</returns>
    public abstract ITensor<TNumber> ScalarMultiply<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;
}