// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// A function parameter.
/// </summary>
[PublicAPI]
public readonly struct Parameter : IIrWritable, IValueEquatable<Parameter>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Parameter"/> struct.
    /// </summary>
    /// <param name="type">The parameter type.</param>
    /// <param name="identifier">The parameter name.</param>
    public Parameter(IrType type, string identifier)
    {
        Type = type;
        Identifier = identifier;
    }

    /// <summary>
    /// Gets the parameter type.
    /// </summary>
    public IrType Type { get; }

    /// <summary>
    /// Gets the parameter name.
    /// </summary>
    public string Identifier { get; }

    /// <inheritdoc />
    public static bool operator ==(Parameter left, Parameter right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Parameter left, Parameter right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Parameter other)
    {
        return Type.Equals(other.Type) && Identifier == other.Identifier;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Parameter other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Type, Identifier);
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        return builder.AppendWritable(Type).Append(' ').Append(Identifier);
    }
}