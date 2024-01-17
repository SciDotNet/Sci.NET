// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents a MSIL switch targets operand.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct MsilSwitchTargetsOperand : IMsilOperand, IValueEquatable<MsilSwitchTargetsOperand>
{
    /// <inheritdoc />
    public required OperandType OperandType { get; init; }

    /// <summary>
    /// Gets the branches.
    /// </summary>
    public required int[] Branches { get; init; }

    /// <summary>
    /// Gets the base offset.
    /// </summary>
    public required int BaseOffset { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MsilSwitchTargetsOperand left, MsilSwitchTargetsOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilSwitchTargetsOperand left, MsilSwitchTargetsOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MsilSwitchTargetsOperand other)
    {
        return OperandType == other.OperandType && Branches.Equals(other.Branches) && BaseOffset == other.BaseOffset;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilSwitchTargetsOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine((int)OperandType, Branches, BaseOffset);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var operand = this;
        return string.Join(", ", Branches.Select(x => $"IL_{x + operand.BaseOffset:X4}"));
    }
}