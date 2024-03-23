// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// Represents a PDB local scope.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required keyword.")]
public readonly struct PdbLocalScope : IValueEquatable<PdbLocalScope>
{
    /// <summary>
    /// Gets the start offset.
    /// </summary>
    public int StartOffset { get; init; }

    /// <summary>
    /// Gets the end offset.
    /// </summary>
    public int EndOffset { get; init; }

    /// <summary>
    /// Gets the length.
    /// </summary>
    public int Length => EndOffset - StartOffset;

    /// <summary>
    /// Gets the variables.
    /// </summary>
    public ImmutableArray<PdbLocalVariable> Variables { get; init; }

    /// <inheritdoc />
    public static bool operator ==(PdbLocalScope left, PdbLocalScope right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(PdbLocalScope left, PdbLocalScope right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(PdbLocalScope other)
    {
        return StartOffset == other.StartOffset && EndOffset == other.EndOffset && Variables.Equals(other.Variables);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is PdbLocalScope other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(StartOffset, EndOffset, Variables);
    }
}