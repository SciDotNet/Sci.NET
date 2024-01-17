// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents a MSIL inline long operand.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct MsilInlineLongOperand : IMsilInlineNumberOperand<long>, IValueEquatable<MsilInlineLongOperand>
{
    /// <inheritdoc />
    public OperandType OperandType => OperandType.InlineI8;

    /// <inheritdoc />
    public required long Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MsilInlineLongOperand left, MsilInlineLongOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilInlineLongOperand left, MsilInlineLongOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MsilInlineLongOperand other)
    {
        return Value == other.Value;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilInlineLongOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Value.ToString("X", CultureInfo.CurrentCulture);
    }
}