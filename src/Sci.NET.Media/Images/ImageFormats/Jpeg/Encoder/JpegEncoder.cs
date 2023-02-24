// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Media.Images.PixelFormats;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg.Encoder;

/// <summary>
/// An encoder for JPEG images.
/// </summary>
[PublicAPI]
public class JpegEncoder : IImageEncoder
{
    /// <inheritdoc />
    public void Encode<TPixel>(Image<TPixel> image, Stream stream)
        where TPixel : unmanaged, IPixel<TPixel>
    {
    }
}