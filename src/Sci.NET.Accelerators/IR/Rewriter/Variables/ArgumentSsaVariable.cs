// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents an argument SSA operand.
/// </summary>
[PublicAPI]
public readonly struct ArgumentSsaVariable : ISsaVariable, IValueEquatable<ArgumentSsaVariable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ArgumentSsaVariable"/> struct.
    /// </summary>
    /// <param name="index">The index of the argument.</param>
    /// <param name="name">The name of the argument.</param>
    /// <param name="type">The type of the argument.</param>
    public ArgumentSsaVariable(int index, string name, Type type)
    {
        Index = index;
        Name = name;
        Type = type;
    }

    /// <summary>
    /// Gets the index of the argument.
    /// </summary>
    public int Index { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Argument;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public static bool operator ==(ArgumentSsaVariable left, ArgumentSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(ArgumentSsaVariable left, ArgumentSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Name}, {Type}";
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(ArgumentSsaVariable other)
    {
        return Index == other.Index && Name == other.Name && Type == other.Type;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is ArgumentSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Index, Name, Type);
    }
}