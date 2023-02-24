// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Media.Images.ColourSpaces;

namespace Sci.NET.Media.Images.PixelFormats;

/// <summary>
/// A 24-bit RGB pixel.
/// </summary>
[PublicAPI]
[StructLayout(LayoutKind.Explicit)]
public readonly struct Rgb24 : IPixel<Rgb24>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Rgb24"/> struct.
    /// </summary>
    /// <param name="red">The red component.</param>
    /// <param name="green">The green component.</param>
    /// <param name="blue">The blue component.</param>
    public Rgb24(byte red, byte green, byte blue)
    {
        Red = red;
        Green = green;
        Blue = blue;
    }

    /// <summary>
    /// Gets the red component of the pixel.
    /// </summary>
    [field: FieldOffset(0)]
    public byte Red { get; }

    /// <summary>
    /// Gets the green component of the pixel.
    /// </summary>
    [field: FieldOffset(1)]
    public byte Green { get; }

    /// <summary>
    /// Gets the blue component of the pixel.
    /// </summary>
    [field: FieldOffset(2)]
    public byte Blue { get; }

    /// <inheritdoc />
    public static bool operator ==(Rgb24 left, Rgb24 right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Rgb24 left, Rgb24 right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc />
    public byte[] ToByteArray()
    {
        return new byte[]
        {
            Red, Green, Blue
        };
    }

    /// <inheritdoc />
    public Rgb24 ToRgb24()
    {
        return this;
    }

    /// <inheritdoc />
    public Rgba32 ToRgba32()
    {
        return new Rgba32(Red, Green, Blue, byte.MaxValue);
    }

    /// <inheritdoc />
    public L8 ToL8()
    {
        return ColourSpaceConverters.Get8BitBt709LuminanceFromRgb24(this);
    }

    /// <inheritdoc />
    public YCbCr ToYCbCr()
    {
        return ColourSpaceConverters.GetYCbCrBt709FromRgb24(Red, Green, Blue);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Rgb24 other)
    {
        return Red == other.Red && Green == other.Green && Blue == other.Blue;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Rgb24 other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Red, Green, Blue);
    }
}