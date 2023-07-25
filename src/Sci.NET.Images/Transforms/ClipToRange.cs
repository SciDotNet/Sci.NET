// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Images.Transforms;

/// <summary>
/// Clips pixel values to a range.
/// </summary>
/// <typeparam name="TNumber">The number type of the image.</typeparam>
public class ClipToRange<TNumber> : IImageTransform<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ClipToRange{TNumber}"/> class.
    /// </summary>
    /// <param name="min">The minimum value to clip to.</param>
    /// <param name="max">The maximum value to clip to.</param>
    public ClipToRange(TNumber min, TNumber max)
    {
        Min = min;
        Max = max;
    }

    /// <summary>
    /// Gets the minimum value to clip to.
    /// </summary>
    public TNumber Min { get; }

    /// <summary>
    /// Gets the maximum value to clip to.
    /// </summary>
    public TNumber Max { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Execute(ITensor<TNumber> tensor)
    {
        return tensor.Clip(Min, Max);
    }
}