// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public readonly struct Int8ConstantSsaVariable : IConstantSsaVariable<long>, IValueEquatable<Int8ConstantSsaVariable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Int8ConstantSsaVariable"/> struct.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Int8ConstantSsaVariable(string name, long value)
    {
        Name = name;
        Value = value;
        Type = typeof(long);
    }

    /// <inheritdoc />
    public long Value { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Constant;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public static bool operator ==(Int8ConstantSsaVariable left, Int8ConstantSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Int8ConstantSsaVariable left, Int8ConstantSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Value}, {Type}";
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Int8ConstantSsaVariable other)
    {
        return Value == other.Value && Name == other.Name && Type == other.Type;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Int8ConstantSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Value, Name, Type);
    }
}