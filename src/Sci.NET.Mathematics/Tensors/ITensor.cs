// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.ReferenceCounting;
using Sci.NET.Mathematics.BLAS.Layout;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface for a tensor, which is an immutable N-Dimensional array.
/// </summary>
/// <typeparam name="TNumber">The type of number stored by the tensor.</typeparam>
[PublicAPI]
public interface ITensor<TNumber> : IDisposable, IReferenceCounted
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the memory block storing the tensor data.
    /// </summary>
    public TypedMemoryHandle<TNumber> Handle { get; }

    /// <inheritdoc cref="Shape.Dimensions" />
    public int[] Dimensions { get; }

    /// <inheritdoc cref="Shape.Rank" />
    public int Rank { get; }

    /// <inheritdoc cref="Shape.ElementCount" />
    public long ElementCount { get; }

    /// <inheritdoc cref="Shape.Strides" />
    public long[] Strides { get; }

    /// <inheritdoc cref="Shape.IsScalar"/>
    public bool IsScalar { get; }

    /// <inheritdoc cref="Shape.IsVector"/>
    public bool IsVector { get; }

    /// <inheritdoc cref="Shape.IsMatrix"/>
    public bool IsMatrix { get; }

    /// <summary>
    /// Gets the transpose type of the tensor.
    /// </summary>
    public TransposeType TransposeType { get; }
}