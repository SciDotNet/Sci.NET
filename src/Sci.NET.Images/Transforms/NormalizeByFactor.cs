// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Images.Transforms;

/// <summary>
/// Normalizes pixel values by a factor.
/// </summary>
/// <typeparam name="TNumber">The number type of the image.</typeparam>
[PublicAPI]
public class NormalizeByFactor<TNumber> : IImageTransform<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NormalizeByFactor{TNumber}"/> class.
    /// </summary>
    /// <param name="scale">The factor to scale by.</param>
    public NormalizeByFactor(TNumber scale)
    {
        Scale = scale;
    }

    /// <summary>
    /// Gets the factor to scale by.
    /// </summary>
    public TNumber Scale { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Execute(ITensor<TNumber> tensor)
    {
        using var scale = new Scalar<TNumber>(Scale, tensor.Backend);
        return tensor.Multiply(scale);
    }
}