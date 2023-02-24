// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Media.Images.ImageFormats;

/// <summary>
/// The colour space of a JPEG image.
/// </summary>
[PublicAPI]
public enum JpegColourSpace
{
    /// <summary>
    /// The image is in an unknown colour space.
    /// </summary>
    None = 0,

    /// <summary>
    /// The image is in grayscale.
    /// </summary>
    GrayScale = 1,

    /// <summary>
    /// The image is in YCbCr.
    /// </summary>
    YCbCr = 2,

    /// <summary>
    /// The image is in CMYK.
    /// </summary>
    CMYK = 3,

    /// <summary>
    /// The image is in YCCK.
    /// </summary>
    YCCK = 4,

    /// <summary>
    /// The image is in RGB.
    /// </summary>
    RGB = 5,
}