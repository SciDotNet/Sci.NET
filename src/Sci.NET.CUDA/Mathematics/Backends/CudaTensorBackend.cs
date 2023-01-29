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
    public override ITensor<TNumber> MatrixMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
    {
        throw new PlatformNotSupportedException();
    }
}