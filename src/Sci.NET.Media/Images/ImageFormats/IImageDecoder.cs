// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Media.Images.PixelFormats;

namespace Sci.NET.Media.Images.ImageFormats;

/// <summary>
/// An interface for image decoders.
/// </summary>
[PublicAPI]
public interface IImageDecoder
{
    /// <summary>
    /// Decodes an image from the specified stream.
    /// </summary>
    /// <param name="stream">The stream to decode from.</param>
    /// <typeparam name="TPixel">The type of pixel to convert to.</typeparam>
    /// <returns>The decoded image.</returns>
    public Image<TPixel> Decode<TPixel>(Stream stream)
        where TPixel : unmanaged, IPixel<TPixel>;
}