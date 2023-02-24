// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using Sci.NET.Common.Performance;
using Sci.NET.Media.Images.PixelFormats;

namespace Sci.NET.Media.Images.ColourSpaces;

/// <summary>
/// Provides methods for converting between colour spaces.
/// </summary>
[PublicAPI]
public static class ColourSpaceConverters
{
    /// <summary>
    /// Converts an RGB colour to a BT709 grayscale colour.
    /// </summary>
    /// <param name="pixel">The pixel to convert.</param>
    /// <returns>The <see cref="L8"/> grayscale colour.</returns>
    [MethodImpl(ImplementationOptions.FastPath)]
    public static L8 Get8BitBt709LuminanceFromRgb24(Rgb24 pixel)
    {
        return Get8BitBt709LuminanceFromRgb24(pixel.Red, pixel.Green, pixel.Blue);
    }

    /// <summary>
    /// Converts an RGB colour to a BT709 grayscale colour.
    /// </summary>
    /// <param name="red">The red component.</param>
    /// <param name="green">The green component.</param>
    /// <param name="blue">The blue component.</param>
    /// <returns>The <see cref="L8"/> grayscale colour.</returns>
    public static L8 Get8BitBt709LuminanceFromRgb24(byte red, byte green, byte blue)
    {
        return new L8((byte)(((red * 0.2126f) + (green * 0.7152f) + (blue * 0.0722f)) * 255));
    }

    /// <summary>
    /// Converts an RGBA colour to a BT709 grayscale colour.
    /// </summary>
    /// <param name="pixel">The pixel to convert.</param>
    /// <returns>The <see cref="L8"/> grayscale colour.</returns>
    [MethodImpl(ImplementationOptions.FastPath)]
    public static L8 Get8BitBt709LuminanceFromRgba32(Rgba32 pixel)
    {
        return Get8BitBt709LuminanceFromRgb24(pixel.Red, pixel.Green, pixel.Blue);
    }

    /// <summary>
    /// Converts an RGB colour to a YCbCr colour.
    /// </summary>
    /// <param name="pixel">The pixel to convert.</param>
    /// <returns>The YCbCr pixel to convert.</returns>
    public static YCbCr GetYCbCrBt709FromRgb24(Rgb24 pixel)
    {
        return GetYCbCrBt709FromRgb24(pixel.Red, pixel.Green, pixel.Blue);
    }

    /// <summary>
    /// Converts an RGB colour to a YCbCr colour.
    /// </summary>
    /// <param name="red">The red component.</param>
    /// <param name="green">The green component.</param>
    /// <param name="blue">The blue component.</param>
    /// <returns>The converted pixel.</returns>
    [MethodImpl(ImplementationOptions.FastPath)]
    public static YCbCr GetYCbCrBt709FromRgb24(byte red, byte green, byte blue)
    {
        var y = (byte)(((red * 0.299f) + (green * 0.587f) + (blue * 0.114f)) * 255);
        var cb = (byte)((((blue - y) * 0.564f) + 128) * 255);
        var cr = (byte)((((red - y) * 0.713f) + 128) * 255);

        return new YCbCr(y, cb, cr);
    }

    /// <summary>
    /// Converts an RGB colour to a YCbCr colour.
    /// </summary>
    /// <param name="pixel">The pixel to convert.</param>
    /// <returns>The converted pixel.</returns>
    public static Rgb24 GetRgb24FromYCbCrBt709(YCbCr pixel)
    {
        return GetRgb24FromYCbCrBt709(pixel.Y, pixel.Cb, pixel.Cr);
    }

    /// <summary>
    /// Converts an RGB colour to a YCbCr colour.
    /// </summary>
    /// <param name="y">The luminance component.</param>
    /// <param name="cb">The blue chrominance component.</param>
    /// <param name="cr">The red chrominance component.</param>
    /// <returns>The converted pixel.</returns>
    public static Rgb24 GetRgb24FromYCbCrBt709(byte y, byte cb, byte cr)
    {
        var r = (byte)(y + (1.403f * (cr - 128)));
        var g = (byte)(y - (0.344f * (cb - 128)) - (0.714f * (cr - 128)));
        var b = (byte)(y + (1.770f * (cb - 128)));

        return new Rgb24(r, g, b);
    }
}