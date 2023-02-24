// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Media.Images.ImageFormats.Jpeg.Markers;

internal readonly struct JFifMarker
{
    private static readonly byte[] JFifIdentifier = "JFIF\0"u8.ToArray();

    public JFifMarker(
        byte majorVersion,
        byte minorVersion,
        byte densityUnits,
        ushort xDensity,
        ushort yDensity,
        byte thumbnailWidth,
        byte thumbnailHeight,
        byte[] thumbnailData)
    {
        MajorVersion = majorVersion;
        MinorVersion = minorVersion;
        DensityUnits = densityUnits;
        XDensity = xDensity;
        YDensity = yDensity;
        ThumbnailWidth = thumbnailWidth;
        ThumbnailHeight = thumbnailHeight;
        ThumbnailData = thumbnailData;
    }

    public byte MajorVersion { get; }

    public byte MinorVersion { get; }

    public byte DensityUnits { get; }

    public ushort XDensity { get; }

    public ushort YDensity { get; }

    public byte ThumbnailWidth { get; }

    public byte ThumbnailHeight { get; }

    public byte[] ThumbnailData { get; }

    public static bool TryParse(Span<byte> bytes, out JFifMarker marker)
    {
        if (bytes == JFifIdentifier)
        {
            marker = new JFifMarker(
                majorVersion: bytes[5],
                minorVersion: bytes[6],
                densityUnits: bytes[7],
                xDensity: (ushort)((bytes[8] << 8) | bytes[9]),
                yDensity: (ushort)((bytes[10] << 8) | bytes[11]),
                thumbnailWidth: bytes[12],
                thumbnailHeight: bytes[13],
                thumbnailData: bytes[14..].ToArray());
            return true;
        }

        marker = default;
        return false;
    }
}