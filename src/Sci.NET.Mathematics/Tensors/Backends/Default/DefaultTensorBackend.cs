// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory.Unmanaged;
using Sci.NET.Mathematics.BLAS;
using Sci.NET.Mathematics.BLAS.Managed;
using Sci.NET.Mathematics.Tensors.Backends.Default.Ops.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.Backends.Default.Ops.Pointwise;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

/// <summary>
/// A managed implementation of <see cref="TensorBackend"/>.
/// </summary>
[PublicAPI]
public class DefaultTensorBackend : TensorBackend
{
    /// <inheritdoc />
    public override INativeMemoryManager MemoryManager => new DefaultNativeMemoryManager();

    /// <inheritdoc />
    public override IBlasProvider BlasProvider => new ManagedBlasProvider();

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
}