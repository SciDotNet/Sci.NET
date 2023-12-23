// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an operand with a byte value.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct InlineByteOperand : IInlineNumberOperand<byte>, IValueEquatable<InlineByteOperand>
{
    /// <summary>
    /// Gets the operand type.
    /// </summary>
    public OperandType OperandType => OperandType.ShortInlineI;

    /// <inheritdoc />
    public required byte Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(InlineByteOperand left, InlineByteOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(InlineByteOperand left, InlineByteOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(InlineByteOperand other)
    {
        return Value == other.Value;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is InlineByteOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"0x{Value:X2}";
    }
}