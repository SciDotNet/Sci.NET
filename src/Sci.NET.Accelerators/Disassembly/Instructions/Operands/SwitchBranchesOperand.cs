// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Instructions.Operands;

/// <summary>
/// Represents a switch branch operand.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct SwitchBranchesOperand : IOperand, IValueEquatable<SwitchBranchesOperand>
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
    public static bool operator ==(SwitchBranchesOperand left, SwitchBranchesOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(SwitchBranchesOperand left, SwitchBranchesOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(SwitchBranchesOperand other)
    {
        return OperandType == other.OperandType && Branches.Equals(other.Branches) && BaseOffset == other.BaseOffset;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is SwitchBranchesOperand other && Equals(other);
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