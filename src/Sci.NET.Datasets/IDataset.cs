// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Datasets.Batches;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Datasets;

/// <summary>
/// An interface for a dataset.
/// </summary>
/// <typeparam name="TBatch">The data type for the batch.</typeparam>
[PublicAPI]
public interface IDataset<out TBatch> : IEnumerable<TBatch>, IDisposable
    where TBatch : IBatch
{
    /// <summary>
    /// Gets the size of each batch.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Gets the number of batches in the dataset.
    /// </summary>
    public int NumBatches { get; }

    /// <summary>
    /// Gets the number of examples in the dataset.
    /// </summary>
    public int NumExamples { get; }

    /// <summary>
    /// Gets the shape of the input.
    /// </summary>
    public Shape InputShape { get; }

    /// <summary>
    /// Gets the shape of the output.
    /// </summary>
    public Shape OutputShape { get; }

    /// <summary>
    /// Shuffles the dataset.
    /// </summary>
    /// <param name="seed">The random seed.</param>
    public void Shuffle(int? seed = null);

    /// <summary>
    /// Gets the next batch.
    /// </summary>
    /// <returns>The next batch.</returns>
    public TBatch NextBatch();
}