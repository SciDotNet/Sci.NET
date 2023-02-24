// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Media.Images.ColourSpaces;

namespace Sci.NET.Media.Images.PixelFormats;

/// <summary>
/// A 32-bit RGBA pixel.
/// </summary>
[PublicAPI]
[StructLayout(LayoutKind.Explicit)]
public readonly struct Rgba32 : IPixel<Rgba32>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Rgba32"/> struct.
    /// </summary>
    /// <param name="red">The red component.</param>
    /// <param name="green">The green component.</param>
    /// <param name="blue">The blue component.</param>
    /// <param name="alpha">The alpha component.</param>
    public Rgba32(byte red, byte green, byte blue, byte alpha)
    {
        Red = red;
        Green = green;
        Blue = blue;
        Alpha = alpha;
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

    /// <summary>
    /// Gets the blue component of the pixel.
    /// </summary>
    [field: FieldOffset(3)]
    public byte Alpha { get; }

    /// <inheritdoc />
    public static bool operator ==(Rgba32 left, Rgba32 right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Rgba32 left, Rgba32 right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc />
    public byte[] ToByteArray()
    {
        return new byte[]
        {
            Red, Green, Blue, Alpha
        };
    }

    /// <inheritdoc />
    public Rgb24 ToRgb24()
    {
        return new Rgb24(Red, Green, Blue);
    }

    /// <inheritdoc />
    public Rgba32 ToRgba32()
    {
        return this;
    }

    /// <inheritdoc />
    public L8 ToL8()
    {
        return ColourSpaceConverters.Get8BitBt709LuminanceFromRgba32(this);
    }

    /// <inheritdoc />
    public YCbCr ToYCbCr()
    {
        return ColourSpaceConverters.GetYCbCrBt709FromRgb24(Red, Green, Blue);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Rgba32 other)
    {
        return Red == other.Red && Green == other.Green && Blue == other.Blue && Alpha == other.Alpha;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Rgba32 other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Red, Green, Blue, Alpha);
    }
}