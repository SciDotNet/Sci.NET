// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Media.Images.PixelFormats;

/// <summary>
/// An interface for pixel types.
/// </summary>
/// <typeparam name="TPixel">The type of the pixel.</typeparam>
[PublicAPI]
public interface IPixel<TPixel> : IValueEquatable<TPixel>
    where TPixel : unmanaged, IPixel<TPixel>
{
    /// <summary>
    /// Gets a byte array representation of the pixel.
    /// </summary>
    /// <returns>A byte array representing the pixels.</returns>
    public byte[] ToByteArray();

    /// <summary>
    /// Converts the pixel to a <see cref="Rgb24"/> pixel.
    /// </summary>
    /// <returns>The pixel in <see cref="Rgb24"/> format.</returns>
    public Rgb24 ToRgb24();

    /// <summary>
    /// Converts the pixel to a <see cref="Rgba32"/> pixel.
    /// </summary>
    /// <returns>The pixel in <see cref="Rgba32"/> format.</returns>
    public Rgba32 ToRgba32();

    /// <summary>
    /// Converts the pixel to an <see cref="L8"/> pixel.
    /// </summary>
    /// <returns>The pixel in <see cref="L8"/> format.</returns>
    public L8 ToL8();

    /// <summary>
    /// Converts the pixel to a <see cref="YCbCr"/> pixel.
    /// </summary>
    /// <returns>The pixel in <see cref="YCbCr"/> format.</returns>
    public YCbCr ToYCbCr();
}