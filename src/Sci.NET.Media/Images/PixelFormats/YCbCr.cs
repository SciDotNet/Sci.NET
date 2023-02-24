// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Media.Images.ColourSpaces;

namespace Sci.NET.Media.Images.PixelFormats;

/// <summary>
/// Represents a YCbCr pixel.
/// </summary>
[PublicAPI]
[StructLayout(LayoutKind.Explicit)]
public readonly struct YCbCr : IPixel<YCbCr>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="YCbCr"/> struct.
    /// </summary>
    /// <param name="y">The luminance component.</param>
    /// <param name="cb">The blue chrominance component.</param>
    /// <param name="cr">The red chrominance component.</param>
    public YCbCr(byte y, byte cb, byte cr)
    {
        Y = y;
        Cb = cb;
        Cr = cr;
    }

    /// <summary>
    /// Gets the Y component.
    /// </summary>
    [field: FieldOffset(0)]
    public byte Y { get; }

    /// <summary>
    /// Gets the Cb component.
    /// </summary>
    [field: FieldOffset(1)]
    public byte Cb { get; }

    /// <summary>
    /// Gets the Cr component.
    /// </summary>
    [field: FieldOffset(2)]
    public byte Cr { get; }

    /// <inheritdoc />
    public static bool operator ==(YCbCr left, YCbCr right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(YCbCr left, YCbCr right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc />
    public byte[] ToByteArray()
    {
        return new byte[]
        {
            Y, Cb, Cr
        };
    }

    /// <inheritdoc />
    public Rgb24 ToRgb24()
    {
        return ColourSpaceConverters.GetRgb24FromYCbCrBt709(this);
    }

    /// <inheritdoc />
    public Rgba32 ToRgba32()
    {
        return ColourSpaceConverters.GetRgb24FromYCbCrBt709(this).ToRgba32();
    }

    /// <inheritdoc />
    public L8 ToL8()
    {
        return new L8(Y);
    }

    /// <inheritdoc />
    public YCbCr ToYCbCr()
    {
        return this;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(YCbCr other)
    {
        return Y == other.Y && Cb == other.Cb && Cr == other.Cr;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is YCbCr other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Y, Cb, Cr);
    }
}