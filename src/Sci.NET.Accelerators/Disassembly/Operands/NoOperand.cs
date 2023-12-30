// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an operand with no value.
/// </summary>
[PublicAPI]
public readonly struct NoOperand : IOperand, IValueEquatable<NoOperand>
{
    /// <inheritdoc />
    public OperandType OperandType => OperandType.InlineNone;

    /// <inheritdoc />
    public static bool operator ==(NoOperand left, NoOperand right)
    {
        _ = left;
        _ = right;
        return true;
    }

    /// <inheritdoc />
    public static bool operator !=(NoOperand left, NoOperand right)
    {
        _ = left;
        _ = right;
        return false;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return string.Empty;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is NoOperand other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(NoOperand other)
    {
        return true;
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return 0;
    }
}