// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents a MSIL branch target operand.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public readonly struct MsilBranchTargetOperand : IMsilOperand, IValueEquatable<MsilBranchTargetOperand>
{
    /// <inheritdoc />
    public required OperandType OperandType { get; init; }

    /// <summary>
    /// Gets the target.
    /// </summary>
    public required int Target { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MsilBranchTargetOperand left, MsilBranchTargetOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilBranchTargetOperand left, MsilBranchTargetOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MsilBranchTargetOperand other)
    {
        return OperandType == other.OperandType && Target == other.Target;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilBranchTargetOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine((int)OperandType, Target);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"IL_{Target:X4}";
    }
}