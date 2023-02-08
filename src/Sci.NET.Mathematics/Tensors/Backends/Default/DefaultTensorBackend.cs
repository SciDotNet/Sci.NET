// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

/// <summary>
/// A managed implementation of <see cref="TensorBackend"/>.
/// </summary>
[PublicAPI]
public class DefaultTensorBackend : TensorBackend
{
    internal const long ParallelizationThreshold = 1000;

    /// <inheritdoc />
    public override IRandomBackendOperations Random { get; } = new DefaultRandomBackendOperations();

    /// <inheritdoc />
    public override ILinearAlgebraBackendOperations LinearAlgebra { get; } =
        new DefaultLinearAlgebraBackendOperations();

    /// <inheritdoc />
    public override ITrigonometryBackendOperations Trigonometry { get; } = new DefaultTrigonometryBackendOperations();

    /// <inheritdoc />
    public override IArithmeticBackendOperations Arithmetic { get; } = new DefaultArithmeticBackendOperations();

    /// <inheritdoc />
    public override IMathematicalBackendOperations MathematicalOperations { get; } =
        new DefaultMathematicalBackendOperations();

    /// <inheritdoc />
    public override IMemoryBlock<TNumber> Create<TNumber>(Shape tensorShape)
    {
        return new SystemMemoryBlock<TNumber>(tensorShape.ElementCount);
    }

    /// <inheritdoc />
    public override void Free<TNumber>(IMemoryBlock<TNumber> handle)
    {
        handle.Dispose();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        if (left.Rank != 0 && right.Rank != 0)
        {
            throw new ArgumentException("At least one of the tensors must be a scalar.");
        }

        var scalar = left.Rank == 0 ? left : right;
        var tensor = left.Rank == 0 ? right : left;

        var result = new Tensor<TNumber>(new Shape(tensor.Dimensions));
        var scalarPtr = scalar.Data[0];
        var tensorPtr = tensor.Data;
        var resultPtr = result.Data;

        for (var i = 0; i < tensor.ElementCount; i++)
        {
            resultPtr[i] = scalarPtr * tensorPtr[i];
        }

        return result;
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(TNumber left, ITensor<TNumber> right)
    {
        var result = new Tensor<TNumber>(new Shape(right.Dimensions));
        var tensorPtr = right.Data;
        var resultPtr = result.Data;

        for (var i = 0; i < right.ElementCount; i++)
        {
            resultPtr[i] = left * tensorPtr[i];
        }

        return result;
    }
}