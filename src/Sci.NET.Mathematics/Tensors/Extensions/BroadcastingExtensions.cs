// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extensions for broadcasting <see cref="ITensor{TNumber}"/> instances.
/// </summary>
[PublicAPI]
public static class BroadcastingExtensions
{
    /// <summary>
    /// Broadcasts a <see cref="ITensor{TNumber}"/> to a new shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to broadcast.</param>
    /// <param name="targetShape">The new shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The broadcast <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Broadcast<TNumber>(this ITensor<TNumber> tensor, Shape targetShape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider.GetTensorOperationServiceProvider()
            .GetBroadcastingService()
            .Broadcast(tensor, targetShape);
    }

    /// <summary>
    /// Determines whether a <see cref="Shape"/> can be broadcast to another <see cref="Shape"/>.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="source">The source <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="target">The target <see cref="Shape"/>.</param>
    /// <returns>A value indicating whether the source <see cref="Shape"/> can be broadcast to the target <see cref="Shape"/>.</returns>
    [DebuggerStepThrough]
    public static bool CanBroadcastTo<TNumber>(this ITensor<TNumber> source, Shape target)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider.GetTensorOperationServiceProvider()
            .GetBroadcastingService()
            .CanBroadcastTo(source.Shape, target);
    }
}