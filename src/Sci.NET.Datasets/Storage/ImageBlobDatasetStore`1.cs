// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Common.Streams;
using Sci.NET.Datasets.Storage.Compression;
using Sci.NET.Images;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Datasets.Storage;

/// <summary>
/// The Image dataset.
/// </summary>
/// <typeparam name="TTensor">The tensor type.</typeparam>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.DocumentationRules", "SA1649:File name should match first type name", Justification = "Non-generic type with same name")]
public class ImageBlobDatasetStore<TTensor> : IDatasetStore<ITensor<TTensor>>
    where TTensor : unmanaged, INumber<TTensor>
{
    private readonly string _fileTablePath;
    private readonly string _blobPath;
    private readonly ConcurrentDictionary<Guid, KeyValuePair<long, int>> _fileTable;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImageBlobDatasetStore{TTensor}"/> class.
    /// </summary>
    /// <param name="blobPath">The path to the blob.</param>
    public ImageBlobDatasetStore(string blobPath)
    {
        var rootPath = Path.GetDirectoryName(blobPath);
        var fileTableName = $"{Path.GetFileNameWithoutExtension(blobPath)}.bin";
        var fileTablePath = Path.Combine(rootPath!, fileTableName);

        _blobPath = blobPath;
        _fileTablePath = fileTablePath;

        var fileTable = new ConcurrentDictionary<Guid, KeyValuePair<long, int>>();
        using var fileTableStream = File.OpenRead(_fileTablePath);

        while (fileTableStream.Position < fileTableStream.Length)
        {
            var id = fileTableStream.ReadValue<Guid>();
            var offset = fileTableStream.ReadValue<long>();
            var length = fileTableStream.ReadValue<int>();

            _ = fileTable.TryAdd(id, new KeyValuePair<long, int>(offset, length));
        }

        _fileTable = fileTable;
    }

    /// <inheritdoc />
    public DatasetCompressionType Compression { get; }

    /// <inheritdoc />
    public ITensor<TTensor> GetItem(Guid id)
    {
        if (!_fileTable.TryGetValue(id, out var offsetLength))
        {
            throw new KeyNotFoundException("The item was not found.");
        }

        using var blobStream = File.OpenRead(_blobPath);
        using var memoryStream = new MemoryStream();

        _ = blobStream.Seek(offsetLength.Key, SeekOrigin.Begin);
        blobStream.CopyTo(memoryStream, offsetLength.Value);

        return Tensor.LoadFromBuffer<TTensor>(memoryStream);
    }

    /// <inheritdoc />
    public Guid AddItem(ITensor<TTensor> item)
    {
        var id = Guid.NewGuid();
        const long offset = 0L;

        using var blobStream = File.Open(_blobPath, FileMode.Append);
        using var memoryStream = new MemoryStream();

        item.Save(memoryStream);

        var length = (int)memoryStream.Length;
        memoryStream.Position = 0;
        memoryStream.CopyTo(blobStream);

        if (!_fileTable.TryAdd(id, new KeyValuePair<long, int>(offset, length)))
        {
            throw new InvalidOperationException("The item could not be added.");
        }

        using var fileTableStream = File.Open(_fileTablePath, FileMode.Append);

        fileTableStream.WriteValue(id);
        fileTableStream.WriteValue(offset);
        fileTableStream.WriteValue(length);

        return id;
    }

    /// <summary>
    /// Adds an image to the dataset.
    /// </summary>
    /// <param name="imagePath">The path to the image.</param>
    /// <returns>The ID of the image.</returns>
    public Guid AddImage(string imagePath)
    {
        using var image = Image
            .LoadRgb24(imagePath)
            .Cast<byte, TTensor>();

        return AddItem(image);
    }
}