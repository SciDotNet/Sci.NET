// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

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
    public void SaveNpy<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>;
}