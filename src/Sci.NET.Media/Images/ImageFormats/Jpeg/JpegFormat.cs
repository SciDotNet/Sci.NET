// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Media.Images.ImageFormats.Jpeg.Decoder;
using Sci.NET.Media.Images.ImageFormats.Jpeg.Encoder;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg;

/// <summary>
/// A format for JPEG images.
/// </summary>
[PublicAPI]
public class JpegFormat : IImageFormat<JpegEncoder, JpegDecoder>
{
    /// <inheritdoc />
    public JpegEncoder Encoder { get; } = new ();

    /// <inheritdoc />
    public JpegDecoder Decoder { get; } = new ();
}