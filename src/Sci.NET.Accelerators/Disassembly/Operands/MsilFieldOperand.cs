// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents a MSIL field operand.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public readonly struct MsilFieldOperand : IMsilOperand, IValueEquatable<MsilFieldOperand>
{
    /// <inheritdoc />
    public required OperandType OperandType { get; init; }

    /// <summary>
    /// Gets the field info.
    /// </summary>
    public required FieldInfo? FieldInfo { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MsilFieldOperand left, MsilFieldOperand right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilFieldOperand left, MsilFieldOperand right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MsilFieldOperand other)
    {
        return OperandType == other.OperandType && EqualityComparer<FieldInfo?>.Default.Equals(FieldInfo, other.FieldInfo);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilFieldOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine((int)OperandType, FieldInfo);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return FieldInfo?.ToString() ?? string.Empty;
    }
}