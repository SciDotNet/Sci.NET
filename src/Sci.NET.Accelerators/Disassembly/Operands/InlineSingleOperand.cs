﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an operand with a float value.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct InlineSingleOperand : IInlineNumberOperand<float>, IValueEquatable<InlineSingleOperand>
{
    /// <inheritdoc />
    public OperandType OperandType => OperandType.ShortInlineR;

    /// <inheritdoc />
    public required float Value { get; init; }

    /// <inheritdoc />
    public static bool operator ==(InlineSingleOperand left, InlineSingleOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(InlineSingleOperand left, InlineSingleOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(InlineSingleOperand other)
    {
        return Value.Equals(other.Value);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is InlineSingleOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Value.ToString("F", CultureInfo.CurrentCulture);
    }
}