// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public readonly struct Int4ConstantSsaVariable : IConstantSsaVariable<int>, IValueEquatable<Int4ConstantSsaVariable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Int4ConstantSsaVariable"/> struct.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Int4ConstantSsaVariable(string name, int value)
    {
        Name = name;
        Value = value;
        Type = typeof(int);
    }

    /// <inheritdoc />
    public int Value { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Constant;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public static bool operator ==(Int4ConstantSsaVariable left, Int4ConstantSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Int4ConstantSsaVariable left, Int4ConstantSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Value}, {Type}";
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Int4ConstantSsaVariable other)
    {
        return Value == other.Value && Name == other.Name && Type == other.Type;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Int4ConstantSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Value, Name, Type);
    }
}