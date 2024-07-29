// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Common;

/// <summary>
/// Represents a three-dimensional size.
/// </summary>
[PublicAPI]
public readonly struct Dim3 : IValueEquatable<Dim3>
{
    /// <summary>
    /// Gets the size in the X dimension.
    /// </summary>
    public int X { get; init; }

    /// <summary>
    /// Gets the size in the Y dimension.
    /// </summary>
    public int Y { get; init; }

    /// <summary>
    /// Gets the size in the Z dimension.
    /// </summary>
    public int Z { get; init; }

    /// <inheritdoc />
    public static bool operator ==(Dim3 left, Dim3 right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Dim3 left, Dim3 right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Dim3 other)
    {
        return X == other.X && Y == other.Y && Z == other.Z;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Dim3 other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(X, Y, Z);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{{{X}, {Y}, {Z}}})";
    }
}