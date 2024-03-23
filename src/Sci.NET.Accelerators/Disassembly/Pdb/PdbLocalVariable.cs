// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// Represents a PDB local variable.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required keyword.")]
public readonly struct PdbLocalVariable : IValueEquatable<PdbLocalVariable>
{
    /// <summary>
    /// Gets the index of the local variable.
    /// </summary>
    public required int Index { get; init; }

    /// <summary>
    /// Gets the name of the local variable.
    /// </summary>
    public required string Name { get; init; }

    /// <inheritdoc />
    public static bool operator ==(PdbLocalVariable left, PdbLocalVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(PdbLocalVariable left, PdbLocalVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(PdbLocalVariable other)
    {
        return Index == other.Index && Name == other.Name;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is PdbLocalVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Index, Name);
    }
}