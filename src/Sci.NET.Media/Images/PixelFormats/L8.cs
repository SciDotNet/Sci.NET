// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Media.Images.ColourSpaces;

namespace Sci.NET.Media.Images.PixelFormats;

/// <summary>
/// A 8-bit luminance pixel.
/// </summary>
[PublicAPI]
[StructLayout(LayoutKind.Explicit)]
public readonly struct L8 : IPixel<L8>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="L8"/> struct.
    /// </summary>
    /// <param name="luminance">The luminance component.</param>
    public L8(byte luminance)
    {
        Luminance = luminance;
    }

    /// <summary>
    /// Gets the luminance value.
    /// </summary>
    [field: FieldOffset(0)]
    public byte Luminance { get; }

    /// <inheritdoc />
    public static bool operator ==(L8 left, L8 right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(L8 left, L8 right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc />
    public byte[] ToByteArray()
    {
        return new byte[]
        {
            Luminance
        };
    }

    /// <inheritdoc />
    public Rgb24 ToRgb24()
    {
        return new Rgb24(Luminance, Luminance, Luminance);
    }

    /// <inheritdoc />
    public Rgba32 ToRgba32()
    {
        return new Rgba32(Luminance, Luminance, Luminance, byte.MaxValue);
    }

    /// <inheritdoc />
    public L8 ToL8()
    {
        return this;
    }

    /// <inheritdoc />
    public YCbCr ToYCbCr()
    {
        return ColourSpaceConverters.GetYCbCrBt709FromRgb24(Luminance, Luminance, Luminance);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(L8 other)
    {
        return Luminance == other.Luminance;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is L8 other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return Luminance.GetHashCode();
    }
}