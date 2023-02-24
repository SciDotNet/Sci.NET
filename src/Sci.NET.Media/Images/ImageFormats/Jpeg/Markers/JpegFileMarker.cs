// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Media.Images.ImageFormats.Jpeg.Markers;

internal readonly struct JpegFileMarker
{
    public JpegFileMarker(byte marker, long position, bool isInvalid)
    {
        IsInvalid = isInvalid;
        Marker = marker;
        Position = position;
    }

    public bool IsInvalid { get; }

    public byte Marker { get; }

    public long Position { get; }
}