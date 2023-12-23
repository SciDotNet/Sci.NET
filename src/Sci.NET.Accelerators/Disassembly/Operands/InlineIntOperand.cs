// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an operand with an int value.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct InlineIntOperand : IInlineNumberOperand<int>, IValueEquatable<InlineIntOperand>
{
    /// <inheritdoc />
    public OperandType OperandType => OperandType.InlineI;

    /// <inheritdoc />
    public required int Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(InlineIntOperand left, InlineIntOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(InlineIntOperand left, InlineIntOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(InlineIntOperand other)
    {
        return Value == other.Value;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is InlineIntOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return Value;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Value.ToString("X", CultureInfo.CurrentCulture);
    }
}