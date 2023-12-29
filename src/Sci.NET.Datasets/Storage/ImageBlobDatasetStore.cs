// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Datasets.Storage;

/// <summary>
/// A dataset store for storing images in a blob.
/// </summary>
[PublicAPI]
public static class ImageBlobDatasetStore
{
    /// <summary>
    /// Creates a new <see cref="ImageBlobDatasetStore{TTensor}"/> at the given path.
    /// </summary>
    /// <param name="blobPath">The path to the blob.</param>
    /// <typeparam name="TNumber">The number type.</typeparam>
    /// <returns>The new <see cref="ImageBlobDatasetStore{TTensor}"/>.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the dataset already exists.</exception>
    public static ImageBlobDatasetStore<TNumber> Create<TNumber>(string blobPath)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var rootPath = Path.GetDirectoryName(blobPath);
        var fileTableName = $"{Path.GetFileNameWithoutExtension(blobPath)}.bin";
        var fileTablePath = Path.Combine(rootPath, fileTableName);

        if (File.Exists(fileTableName))
        {
            throw new InvalidOperationException("The dataset already exists.");
        }

        using var file = File.Create(blobPath);
        using var fileTable = File.Create(fileTablePath);

        file.Close();
        fileTable.Close();

        return new ImageBlobDatasetStore<TNumber>(blobPath);
    }
}