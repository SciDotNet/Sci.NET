// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Media.Images.ImageFormats;

/// <summary>
/// An interface for image formats.
/// </summary>
/// <typeparam name="TEncoder">The type of encoder for the format.</typeparam>
/// <typeparam name="TDecoder">The type of decoder for the format.</typeparam>
[PublicAPI]
public interface IImageFormat<out TEncoder, out TDecoder>
    where TEncoder : IImageEncoder
    where TDecoder : IImageDecoder
{
    /// <summary>
    /// Gets the encoder for the format.
    /// </summary>
    public TEncoder Encoder { get; }

    /// <summary>
    /// Gets the decoder for the format.
    /// </summary>
    public TDecoder Decoder { get; }
}