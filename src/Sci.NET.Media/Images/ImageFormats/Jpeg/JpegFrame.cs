// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Media.Images.ImageFormats.Jpeg.Components;
using Sci.NET.Media.Images.ImageFormats.Jpeg.Markers;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg;

internal class JpegFrame : IDisposable
{
    public bool IsExtended { get; }

    public bool IsProgressive { get; }

    public bool IsInterleaved { get; }

    public byte Precision { get; }

    public float MaxColourValue { get; }

    public int PixelWidth { get; }

    public int PixelHeight { get; }

    public byte ComponentCount { get; }

    public byte ComponentOrder { get; }

    public JpegComponent[] Components { get; }

    public int McusPerLine { get; }

    public int McusPerColumn { get; }

    public int BitsPerPixel => ComponentCount * Precision;

    public JpegFrame(JpegFileMarker sofMarker, byte precision, int width, int height, byte componentCount)
    {
        IsExtended = sofMarker.Marker == JpegMarkers.SOF2;
    }

    ~JpegFrame()
    {
        Dispose(false);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
        }
    }
}