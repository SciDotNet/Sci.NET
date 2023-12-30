// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Datasets.Storage.Compression;

namespace Sci.NET.Datasets.Storage;

/// <summary>
/// An interface for a dataset store.
/// </summary>
/// <typeparam name="TItem">The type of the item.</typeparam>
[PublicAPI]
public interface IDatasetStore<TItem>
{
    /// <summary>
    /// Gets the compression type.
    /// </summary>
    public DatasetCompressionType Compression { get; }

    /// <summary>
    /// Gets a stream for the item with the given id.
    /// </summary>
    /// <param name="id">The Id of the item.</param>
    /// <returns>A stream for the item.</returns>
    public TItem GetItem(Guid id);

    /// <summary>
    /// Adds an item to the dataset.
    /// </summary>
    /// <param name="item">The item to add.</param>
    /// <returns>The Id of the item.</returns>
    public Guid AddItem(TItem item);
}