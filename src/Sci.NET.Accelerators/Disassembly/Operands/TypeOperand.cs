// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an operand with a type value.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct TypeOperand : IOperand, IValueEquatable<TypeOperand>
{
    /// <inheritdoc />
    public required OperandType OperandType { get; init; }

    /// <summary>
    /// Gets the operand value.
    /// </summary>
    public required Type Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(TypeOperand left, TypeOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(TypeOperand left, TypeOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(TypeOperand other)
    {
        return OperandType == other.OperandType && Value == other.Value;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is TypeOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine((int)OperandType, Value);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Value?.ToString() ?? string.Empty;
    }
}