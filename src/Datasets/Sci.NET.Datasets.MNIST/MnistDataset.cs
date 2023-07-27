// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Numerics;
using Sci.NET.Common.Random;
using Sci.NET.Datasets.Batches;
using Sci.NET.Images.Transforms;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Datasets.MNIST;

/// <summary>
/// A dataset for the MNIST handwritten digit recognition problem.
/// </summary>
/// <typeparam name="TNumber">The number type of the dataset.</typeparam>
[PublicAPI]
public class MnistDataset<TNumber> : IDataset<ImageOneHotCategoryBatch<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Tensor<TNumber> _trainingImages;
    private readonly Matrix<TNumber> _trainingLabels;
    private readonly List<int> _sampleOrder;
    private int _currentBatch;

    /// <summary>
    /// Initializes a new instance of the <see cref="MnistDataset{TNumber}"/> class.
    /// </summary>
    /// <param name="batchSize">The size of each batch.</param>
    /// <param name="transforms">A collection of transforms to apply to the input images.</param>
    public MnistDataset(int batchSize, params IImageTransform<TNumber>[] transforms)
    {
        var root = Path.GetDirectoryName(typeof(MnistDataset<TNumber>).Assembly.Location);

        var trainingImages = Tensor.Load<byte>($@"{root}\resources\images.sdnt");
        var trainingLabels = Tensor.Load<byte>($@"{root}\resources\labels.sdnt");

        _trainingImages = trainingImages.ToTensor().Cast<byte, TNumber>();
        _trainingLabels = ((ITensor<TNumber>)trainingLabels.ToTensor().Cast<byte, TNumber>()).ToMatrix();

        Transforms = transforms;
        BatchSize = batchSize;

        var inputShape = _trainingImages.Shape.ToArray();
        inputShape[0] = BatchSize;

        InputShape = new Shape(inputShape);
        OutputShape = new Shape(BatchSize, _trainingLabels.Shape[1]);

        _sampleOrder = new List<int>();

        for (var i = 0; i < (BatchSize * NumBatches) - 1; i++)
        {
            _sampleOrder.Add(i % NumExamples);
        }
    }

    /// <inheritdoc />
    public int BatchSize { get; }

    /// <inheritdoc />
    public int NumBatches => (int)Math.Ceiling(_trainingImages.Shape[0] / (double)BatchSize);

    /// <inheritdoc />
    public int NumExamples => _trainingImages.Shape[0];

    /// <summary>
    /// Gets the transforms applied to the dataset.
    /// </summary>
    public IReadOnlyCollection<IImageTransform<TNumber>> Transforms { get; }

    /// <inheritdoc />
    public Shape InputShape { get; }

    /// <inheritdoc />
    public Shape OutputShape { get; }

    /// <summary>
    /// Gets or sets a value indicating whether the dataset should be shuffled.
    /// </summary>
    public bool ShouldShuffle { get; set; } = true;

    /// <inheritdoc />
    public void Shuffle(int? seed = null)
    {
        _sampleOrder.Shuffle(seed);
    }

    /// <inheritdoc />
    public ImageOneHotCategoryBatch<TNumber> NextBatch()
    {
        if (_currentBatch >= NumBatches)
        {
            throw new InvalidOperationException("No more batches to return.");
        }

        _currentBatch++;

        var batchImages = new List<Tensor<TNumber>>();
        var batchLabels = new List<Mathematics.Tensors.Vector<TNumber>>();
        var nextIndices = new List<int>();
        var batchStartIndex = _currentBatch * BatchSize;

        for (var i = 0; i < BatchSize; i++)
        {
            nextIndices.Add(_sampleOrder[batchStartIndex + i]);
        }

        foreach (var nextExample in nextIndices)
        {
            var image = _trainingImages[nextExample];
            var label = _trainingLabels[nextExample];

            foreach (var transform in Transforms)
            {
                var newImage = transform.Execute(image);
                image.Dispose();
                image = newImage;
            }

            batchImages.Add(image.ToTensor());
            batchLabels.Add(label);
        }

        var result = new ImageOneHotCategoryBatch<TNumber>(batchImages.Concatenate(), batchLabels.Concatenate());

        foreach (var tmpTensor in batchImages.Concat<ITensor<TNumber>>(batchLabels))
        {
            tmpTensor.Dispose();
        }

        return result;
    }

    /// <inheritdoc />
    public IEnumerator<ImageOneHotCategoryBatch<TNumber>> GetEnumerator()
    {
        if (ShouldShuffle)
        {
            Shuffle();
        }

        _currentBatch = 0;

        while (HasNextBatch())
        {
            yield return NextBatch();
        }
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the dataset.
    /// </summary>
    /// <param name="disposing">Whether or not to dispose of managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _trainingImages.Dispose();
            _trainingLabels.Dispose();
        }
    }

    private bool HasNextBatch()
    {
        return _currentBatch < NumBatches;
    }
}