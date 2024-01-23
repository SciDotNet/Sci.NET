// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a local variable.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public readonly struct LocalVariable : IValueEquatable<LocalVariable>
{
    /// <summary>
    /// Gets the name of the local variable.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the type of the local variable.
    /// </summary>
    public required Type Type { get; init; }

    /// <summary>
    /// Gets the index of the local variable.
    /// </summary>
    public required int Index { get; init; }

    /// <inheritdoc />
    public static bool operator ==(LocalVariable left, LocalVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(LocalVariable left, LocalVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(LocalVariable other)
    {
        return Name == other.Name && Type == other.Type && Index == other.Index;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is LocalVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Name, Type, Index);
    }
}