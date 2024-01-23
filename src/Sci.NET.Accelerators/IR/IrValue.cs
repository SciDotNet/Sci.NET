// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// A value in the intermediate representation.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public readonly struct IrValue : IIrWritable, IValueEquatable<IrValue>
{
    /// <summary>
    /// Gets the value type.
    /// </summary>
    public required IrType Type { get; init; }

    /// <summary>
    /// Gets the value identifier.
    /// </summary>
    public required string Identifier { get; init; }

    /// <inheritdoc />
    public static bool operator ==(IrValue left, IrValue right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(IrValue left, IrValue right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(IrValue other)
    {
        return Type.Equals(other.Type) && Identifier == other.Identifier;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is IrValue other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Type, Identifier);
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        if (Type.IsPointer)
        {
            return builder.AppendWritable(Type).Append(' ').Append(Identifier);
        }

        return builder.AppendWritable(Type).Append(' ').Append('%').Append(Identifier);
    }
}