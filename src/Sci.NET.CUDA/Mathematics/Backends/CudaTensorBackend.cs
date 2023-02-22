// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.CUDA.Mathematics.Backends;

/// <summary>
/// A CUDA implementation of the <see cref="TensorBackend"/> interface.
/// </summary>
[PublicAPI]
public class CudaTensorBackend : TensorBackend
{
    /// <inheritdoc />
    public override IRandomBackendOperations Random => throw new PlatformNotSupportedException();

    /// <inheritdoc />
    public override ILinearAlgebraBackendOperations LinearAlgebra => throw new PlatformNotSupportedException();

    /// <inheritdoc />
    public override ITrigonometryBackendOperations Trigonometry => throw new PlatformNotSupportedException();

    /// <inheritdoc />
    public override IArithmeticBackendOperations Arithmetic => throw new PlatformNotSupportedException();

    /// <inheritdoc />
    public override IMathematicalBackendOperations MathematicalOperations => throw new PlatformNotSupportedException();

    /// <inheritdoc />
    public override INeuralNetworkBackendOperations NeuralNetwork => throw new PlatformNotSupportedException();

    /// <inheritdoc />
    public override IMemoryBlock<TNumber> Create<TNumber>(Shape tensorShape)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override void Free<TNumber>(IMemoryBlock<TNumber> handle)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(TNumber left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }
}