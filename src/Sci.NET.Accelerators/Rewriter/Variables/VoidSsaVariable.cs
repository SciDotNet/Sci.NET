// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Rewriter.Variables;

/// <summary>
/// Represents a void operand used to represent an operation with no result.
/// </summary>
[PublicAPI]
public readonly struct VoidSsaVariable : ISsaVariable, IValueEquatable<VoidSsaVariable>
{
    /// <inheritdoc />
    public string Name => string.Empty;

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.None;

    /// <inheritdoc />
    public Type Type => typeof(void);

    /// <inheritdoc />
    public static bool operator ==(VoidSsaVariable left, VoidSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(VoidSsaVariable left, VoidSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(VoidSsaVariable other)
    {
        return true;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is VoidSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Name, Type, DeclarationType);
    }
}