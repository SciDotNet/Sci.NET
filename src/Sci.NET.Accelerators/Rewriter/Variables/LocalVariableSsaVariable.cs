// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Rewriter.Variables;

/// <summary>
/// Represents a local variable SSA operand.
/// </summary>
[PublicAPI]
public readonly struct LocalVariableSsaVariable : ISsaVariable, IValueEquatable<LocalVariableSsaVariable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LocalVariableSsaVariable"/> struct.
    /// </summary>
    /// <param name="index">The index of the local variable.</param>
    /// <param name="name">The name of the local variable.</param>
    /// <param name="type">The type of the local variable.</param>
    public LocalVariableSsaVariable(int index, string name, Type type)
    {
        Index = index;
        Name = name;
        Type = type;
    }

    /// <summary>
    /// Gets the index of the local variable.
    /// </summary>
    public int Index { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Local;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public static bool operator ==(LocalVariableSsaVariable left, LocalVariableSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(LocalVariableSsaVariable left, LocalVariableSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Name}, {Type}";
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(LocalVariableSsaVariable other)
    {
        return Index == other.Index && Name == other.Name && Type == other.Type;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is LocalVariableSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Index, Name, Type);
    }
}