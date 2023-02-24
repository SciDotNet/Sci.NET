// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Media.Images.PixelFormats;

namespace Sci.NET.Media.Images.ImageFormats;

/// <summary>
/// An interface for image encoders.
/// </summary>
[PublicAPI]
public interface IImageEncoder
{
    /// <summary>
    /// Encodes the specified image to the specified stream.
    /// </summary>
    /// <param name="image">The image to encode.</param>
    /// <param name="stream">The stream to encode to.</param>
    /// <typeparam name="TPixel">The type of pixel to write.</typeparam>
    public void Encode<TPixel>(Image<TPixel> image, Stream stream)
        where TPixel : unmanaged, IPixel<TPixel>;
}