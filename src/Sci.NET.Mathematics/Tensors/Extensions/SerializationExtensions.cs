// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

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
    public static void SaveNumpy<TNumber>(this ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ServiceProvider
            .GetTensorOperationServiceProvider()
            .GetSerializationService()
            .SaveNpy(tensor, path);
    }
}