// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Extensions;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Extension methods for <see cref="ITensor{T}"/>.
/// </summary>
[PublicAPI]
public static class TensorExtensions
{
    /// <summary>
    /// Converts a tensor to an array.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to convert to an array.</param>
    /// <typeparam name="T">The type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <see cref="ITensor{TNumber}"/> data as an array.</returns>
    public static T[] ToArray<T>(this ITensor<T> tensor)
        where T : unmanaged, INumber<T>
    {
        return tensor.Data.ToArray();
    }

    /// <summary>
    /// Saves a <see cref="ITensor{TNumber}"/> to a file.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to save.</param>
    /// <param name="path">The path to save to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public static unsafe void Save<TNumber>(this ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var elementCount = tensor.ElementCount;
        var rank = tensor.Rank;
        var dims = tensor.Dimensions;
        using var data = tensor.Data.ToSystemMemory();
        using var file = File.OpenWrite(path);
        using var unmanagedData = new UnmanagedMemoryStream((byte*)data.ToPointer(), elementCount * sizeof(TNumber));

        file.Write(elementCount);
        file.Write(rank);
        file.Write(dims);
        unmanagedData.CopyTo(file);
        file.Flush();
        file.Close();
    }
}