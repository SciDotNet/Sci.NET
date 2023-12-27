// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public readonly struct Float8ConstantSsaVariable : IConstantSsaVariable<double>, IValueEquatable<Float8ConstantSsaVariable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Float8ConstantSsaVariable"/> struct.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Float8ConstantSsaVariable(string name, double value)
    {
        Name = name;
        Type = typeof(double);
        Value = value;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Constant;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public double Value { get; }

    /// <inheritdoc />
    public static bool operator ==(Float8ConstantSsaVariable left, Float8ConstantSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Float8ConstantSsaVariable left, Float8ConstantSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Value}, {Type}";
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Float8ConstantSsaVariable other)
    {
        return Name == other.Name && Type == other.Type && Value.Equals(other.Value);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Float8ConstantSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Name, Type, Value);
    }
}