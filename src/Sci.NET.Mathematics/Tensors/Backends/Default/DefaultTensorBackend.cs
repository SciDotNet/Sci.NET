// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors.Backends.Default.Ops.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.Backends.Default.Ops.Elementwise;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

/// <summary>
/// A managed implementation of <see cref="TensorBackend"/>.
/// </summary>
[PublicAPI]
public class DefaultTensorBackend : TensorBackend
{
    internal const long ParallelizationThreshold = 1000;

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
    public override ITensor<TNumber> MatrixMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return SumProductOperations.MatrixMultiply(left, right);
    }

    /// <inheritdoc />
    public override ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return SumProductOperations.InnerProduct(left, right);
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return ScalarProductOperations.ScalarProduct(left, right);
    }

    /// <inheritdoc />
    public override ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
    {
        return TrigonometryOperations.Sin(tensor);
    }

    /// <inheritdoc />
    public override ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
    {
        return TrigonometryOperations.Cos(tensor);
    }

    /// <inheritdoc />
    public override ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
    {
        return TrigonometryOperations.Tan(tensor);
    }
}