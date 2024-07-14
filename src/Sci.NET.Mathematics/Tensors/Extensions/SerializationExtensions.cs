// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Attributes;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Provides extension methods for serialization.
/// </summary>
[PublicAPI]
public static class SerializationExtensions
{
    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a file in the npy format.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="path">The path for the numpy file.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    [DebuggerStepThrough]
    [PreviewFeature]
    public static void SaveNumpy<TNumber>(this ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .SaveNpy(tensor, path);
    }

    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a file.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="path">The path for the file.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    [DebuggerStepThrough]
    [PreviewFeature]
    public static void Save<TNumber>(this ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .Save(tensor, path);
    }

    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a file.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="stream">The stream to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    [DebuggerStepThrough]
    [PreviewFeature]
    public static void Save<TNumber>(this ITensor<TNumber> tensor, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .Save(tensor, stream);
    }

    /// <summary>
    /// Saves a dictionary of named <see cref="ITensor{TNumber}"/> to a file in the safetensors format.
    /// </summary>
    /// <param name="tensors">The dictionary of named <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="path">The path of the file to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    [DebuggerStepThrough]
    public static void SaveSafeTensors<TNumber>(this Dictionary<string, ITensor<TNumber>> tensors, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .SaveSafeTensors(tensors, path);
    }

    /// <summary>
    /// Saves a dictionary of named <see cref="ITensor{TNumber}"/> to a stream in the safetensors format.
    /// </summary>
    /// <param name="tensors">The dictionary of named <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="stream">The stream to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public static void SaveSafeTensors<TNumber>(this Dictionary<string, ITensor<TNumber>> tensors, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .SaveSafeTensors(tensors, stream);
    }

    /// <summary>
    /// Compresses and saves a <see cref="ITensor{TNumber}"/> to a file.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to serialize.</param>
    /// <param name="file">The path for the file.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public static void SaveCompressed<TNumber>(this ITensor<TNumber> tensor, string file)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .SaveCompressed(tensor, file);
    }
}