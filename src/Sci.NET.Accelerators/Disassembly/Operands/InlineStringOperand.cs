// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an operand with a string value.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct InlineStringOperand : IOperand, IValueEquatable<InlineStringOperand>
{
    /// <inheritdoc />
    public OperandType OperandType => OperandType.InlineString;

    /// <summary>
    /// Gets the operand value.
    /// </summary>
    public required string Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(InlineStringOperand left, InlineStringOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(InlineStringOperand left, InlineStringOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(InlineStringOperand other)
    {
        return string.Equals(Value, other.Value, StringComparison.Ordinal);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is InlineStringOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return StringComparer.Ordinal.GetHashCode(Value);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Value;
    }
}