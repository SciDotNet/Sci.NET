// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents a method operand.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
[PublicAPI]
public readonly struct MethodOperand : IOperand, IValueEquatable<MethodOperand>
{
    /// <inheritdoc />
    public required OperandType OperandType { get; init; }

    /// <summary>
    /// Gets the method base for the call.
    /// </summary>
    public required MethodBase? MethodBase { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MethodOperand left, MethodOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MethodOperand left, MethodOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MethodOperand other)
    {
        return OperandType == other.OperandType && Equals(MethodBase, other.MethodBase);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MethodOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine((int)OperandType, MethodBase);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return MethodBase?.ToString() ?? string.Empty;
    }
}