// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg.Markers;

internal readonly struct AdobeMarker : IValueEquatable<AdobeMarker>
{
    public const int Length = 12;

    private static readonly byte[] AdobeMarkerBytes = "Adobe"u8.ToArray();

    private AdobeMarker(short dctEncodeVersion, short app14Flags0, short app14Flags1, byte colourTransform)
    {
        DctEncodeVersion = dctEncodeVersion;
        App14Flags0 = app14Flags0;
        App14Flags1 = app14Flags1;
        ColourTransform = colourTransform;
    }

    public byte ColourTransform { get; }

    public short App14Flags1 { get; }

    public short App14Flags0 { get; }

    public short DctEncodeVersion { get; }

    public static bool operator ==(AdobeMarker left, AdobeMarker right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(AdobeMarker left, AdobeMarker right)
    {
        return !left.Equals(right);
    }

    public override bool Equals(object? obj)
    {
        return obj is AdobeMarker other && Equals(other);
    }

    public static bool TryParse(byte[] bytes, out AdobeMarker marker)
    {
        if (bytes.Length >= AdobeMarkerBytes.Length &&
            bytes[..AdobeMarkerBytes.Length].SequenceEqual(AdobeMarkerBytes))
        {
            var dctEncodeVersion = (short)((bytes[5] << 8) | bytes[6]);
            var app14Flags0 = (short)((bytes[7] << 8) | bytes[8]);
            var app14Flags1 = (short)((bytes[9] << 8) | bytes[10]);
            var colourTransform = bytes[11];

            marker = new AdobeMarker(dctEncodeVersion, app14Flags0, app14Flags1, colourTransform);
            return true;
        }

        marker = default;
        return false;
    }

    public bool Equals(AdobeMarker other)
    {
        return ColourTransform == other.ColourTransform &&
               App14Flags1 == other.App14Flags1 &&
               App14Flags0 == other.App14Flags0 &&
               DctEncodeVersion == other.DctEncodeVersion;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(ColourTransform, App14Flags1, App14Flags0, DctEncodeVersion);
    }
}