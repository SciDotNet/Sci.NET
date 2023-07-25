// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// An interface providing methods to broadcast <see cref="ITensor{TNumber}"/> instances.
/// </summary>
[PublicAPI]
public interface IBroadcastService
{
    /// <summary>
    /// Determines whether a <see cref="Shape"/> can be broadcast to another <see cref="Shape"/>.
    /// </summary>
    /// <param name="source">The source <see cref="Shape"/>.</param>
    /// <param name="target">The target <see cref="Shape"/>.</param>
    /// <returns>Whether the source <see cref="Shape"/> can be broadcast to the target <see cref="Shape"/>.</returns>
    public bool CanBroadcastTo(Shape source, Shape target);

    /// <summary>
    /// Determines whether a binary operation can be broadcast.
    /// </summary>
    /// <param name="left">The shape of the left operand.</param>
    /// <param name="right">The shape of the right operand.</param>
    /// <returns>Whether the binary operation can be broadcast.</returns>
    public bool CanBroadcastBinaryOp(Shape left, Shape right);

    /// <summary>
    /// Broadcasts a <see cref="ITensor{TNumber}"/> to a new shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to broadcast.</param>
    /// <param name="targetShape">The new shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The broadcast <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Broadcast<TNumber>(ITensor<TNumber> tensor, Shape targetShape)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Broadcasts two <see cref="ITensor{TNumber}"/> instances to a common shape.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/> to broadcast.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/> to broadcast.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The broadcast <see cref="ITensor{TNumber}"/> instances.</returns>
    public (ITensor<TNumber> Left, ITensor<TNumber> Right) Broadcast<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;
}