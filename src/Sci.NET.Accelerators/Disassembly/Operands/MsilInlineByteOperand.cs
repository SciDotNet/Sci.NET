﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents a MSIL inline byte operand.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public readonly struct MsilInlineByteOperand : IMsilInlineNumberOperand<byte>, IValueEquatable<MsilInlineByteOperand>
{
    /// <summary>
    /// Gets the operand type.
    /// </summary>
    public OperandType OperandType => OperandType.ShortInlineI;

    /// <inheritdoc />
    public required byte Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MsilInlineByteOperand left, MsilInlineByteOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilInlineByteOperand left, MsilInlineByteOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MsilInlineByteOperand other)
    {
        return Value == other.Value;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilInlineByteOperand other && Equals(other);
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