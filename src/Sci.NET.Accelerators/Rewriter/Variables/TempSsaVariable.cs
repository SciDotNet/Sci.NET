// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Rewriter.Variables;

/// <summary>
/// Represents a temporary operand.
/// </summary>
[PublicAPI]
public readonly struct TempSsaVariable : ISsaVariable, IValueEquatable<TempSsaVariable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TempSsaVariable"/> struct.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="type">The type of the operand.</param>
    public TempSsaVariable(string name, Type type)
    {
        Name = name;
        Type = type;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Temp;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public static bool operator ==(TempSsaVariable left, TempSsaVariable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(TempSsaVariable left, TempSsaVariable right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Name}, {Type}";
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(TempSsaVariable other)
    {
        return Name == other.Name && Type == other.Type;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is TempSsaVariable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Name, Type);
    }

    /// <inheritdoc />
    public IrValue ToIrValue()
    {
        return new () { Identifier = Name, Type = Type.ToIrType() };
    }
}