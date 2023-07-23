// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Datasets.Batches;

/// <summary>
/// A batch of images and labels.
/// </summary>
/// <typeparam name="TNumber">The number type of the batch elements.</typeparam>
public class ImageOneHotCategoryBatch<TNumber> : IBatch
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ImageOneHotCategoryBatch{TNumber}"/> class.
    /// </summary>
    /// <param name="images">The batch images.</param>
    /// <param name="labels">The batch labels.</param>
    [SetsRequiredMembers]
    public ImageOneHotCategoryBatch(Tensor<TNumber> images, Matrix<TNumber> labels)
    {
        Images = images;
        Labels = labels;
    }

#pragma warning disable SA1206
    /// <summary>
    /// Gets the batch images.
    /// </summary>
    public required Tensor<TNumber> Images { get; init; }

    /// <summary>
    /// Gets the batch labels.
    /// </summary>
    public required Matrix<TNumber> Labels { get; init; }

#pragma warning restore SA1206

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the <see cref="ImageOneHotCategoryBatch{TNumber}"/>.
    /// </summary>
    /// <param name="disposing">A value indicating whether the object is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Images.Dispose();
            Labels.Dispose();
        }
    }
}