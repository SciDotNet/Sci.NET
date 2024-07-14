// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Attributes;

namespace Sci.NET.Mathematics.Tensors.Serialization;

/// <summary>
/// A service for serializing and deserializing <see cref="ITensor{TNumber}"/> implementations.
/// </summary>
[PublicAPI]
public interface ISerializationService
{
    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a file in the npy format.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="path">The path to the file to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    [PreviewFeature]
    public void SaveNpy<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a file.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="path">The path to the file to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Save<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a stream.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="stream">The stream to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Save<TNumber>(ITensor<TNumber> tensor, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Saves a compressed <see cref="ITensor{TNumber}"/> to a file.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="path">The path to the file to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SaveCompressed<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Saves a compressed <see cref="ITensor{TNumber}"/> to a stream.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="stream">The stream to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SaveCompressed<TNumber>(ITensor<TNumber> tensor, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Loads a <see cref="ITensor{TNumber}"/> from a file.
    /// </summary>
    /// <param name="path">The path to the file to load from.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The deserialized <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Load<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Loads a <see cref="ITensor{TNumber}"/> from a stream.
    /// </summary>
    /// <param name="stream">The stream to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The deserialized <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Load<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Loads a compressed <see cref="ITensor{TNumber}"/> from a file.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="path">The path to the file to load from.</param>
    /// <returns>The deserialized <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> LoadCompressed<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Loads a compressed <see cref="ITensor{TNumber}"/> from a file.
    /// </summary>
    /// <param name="stream">The stream to load from.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The deserialized <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> LoadCompressed<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Loads a named dictionary of <see cref="ITensor{TNumber}"/> from a file in the safetensors format.
    /// </summary>
    /// <param name="path">The path to the file to load from.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A dictionary of the deserialized <see cref="ITensor{TNumber}"/> objects.</returns>
    public Dictionary<string, ITensor<TNumber>> LoadSafeTensors<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Loads a named dictionary of <see cref="ITensor{TNumber}"/> from a stream in the safetensors format.
    /// </summary>
    /// <param name="stream">The stream to load from.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A dictionary of the deserialized <see cref="ITensor{TNumber}"/> objects.</returns>
    public Dictionary<string, ITensor<TNumber>> LoadSafeTensors<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Saves a named dictionary <see cref="ITensor{TNumber}"/> to a file in the safetensors format.
    /// </summary>
    /// <param name="tensors">The <see cref="ITensor{TNumber}"/> objects to serialize.</param>
    /// <param name="path">The path to the file to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SaveSafeTensors<TNumber>(Dictionary<string, ITensor<TNumber>> tensors, string path)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Saves a named dictionary of <see cref="ITensor{TNumber}"/> to a stream in the safetensors format.
    /// </summary>
    /// <param name="tensors">The <see cref="ITensor{TNumber}"/> objects to serialize.</param>
    /// <param name="stream">The stream to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SaveSafeTensors<TNumber>(Dictionary<string, ITensor<TNumber>> tensors, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>;
}