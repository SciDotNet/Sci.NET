// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Media.Images.PixelFormats;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg.Decoder;

/// <summary>
/// A decoder for JPEG images.
/// </summary>
[PublicAPI]
public class JpegDecoder : IImageDecoder
{
    /// <inheritdoc />
    public Image<TPixel> Decode<TPixel>(Stream stream)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        return new Image<TPixel>(10, 10);
    }
}