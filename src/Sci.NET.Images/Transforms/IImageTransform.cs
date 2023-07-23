// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Images.Transforms;

/// <summary>
/// An interface for an image transform.
/// </summary>
/// <typeparam name="TNumber">The number type of the tensor.</typeparam>
[PublicAPI]
public interface IImageTransform<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Executes the transform on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to transform.</param>
    /// <returns>The transformed <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Execute(ITensor<TNumber> tensor);
}