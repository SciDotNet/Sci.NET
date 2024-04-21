// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// Represents a PDB sequence point.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required keyword.")]
public readonly struct PdbSequencePoint : IValueEquatable<PdbSequencePoint>
{
    /// <summary>
    /// Gets the document name.
    /// </summary>
    public required string DocumentName { get; init; }

    /// <summary>
    /// Gets the start line.
    /// </summary>
    public required int StartLine { get; init; }

    /// <summary>
    /// Gets the start column.
    /// </summary>
    public required int StartColumn { get; init; }

    /// <summary>
    /// Gets the end line.
    /// </summary>
    public required int EndLine { get; init; }

    /// <summary>
    /// Gets the end column.
    /// </summary>
    public required int EndColumn { get; init; }

    /// <summary>
    /// Gets the offset.
    /// </summary>
    public required int Offset { get; init; }

    /// <inheritdoc />
    public static bool operator ==(PdbSequencePoint left, PdbSequencePoint right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(PdbSequencePoint left, PdbSequencePoint right)
    {
        return !left.Equals(right);
    }

    /// <summary>
    /// Gets a sequence point that does not exist.
    /// </summary>
    /// <returns>A sequence point that does not exist.</returns>
    public static PdbSequencePoint None()
    {
        return new ()
        {
            Offset = -1,
            DocumentName = string.Empty,
            StartLine = -1,
            StartColumn = -1,
            EndLine = -1,
            EndColumn = -1
        };
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(PdbSequencePoint other)
    {
        return DocumentName == other.DocumentName &&
               StartLine == other.StartLine &&
               StartColumn == other.StartColumn &&
               EndLine == other.EndLine &&
               EndColumn == other.EndColumn &&
               Offset == other.Offset;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is PdbSequencePoint other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(
            DocumentName,
            StartLine,
            StartColumn,
            EndLine,
            EndColumn,
            Offset);
    }
}