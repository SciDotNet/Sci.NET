// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// Represents a variable.
/// </summary>
[PublicAPI]
public readonly struct Variable : IValueEquatable<Variable>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Variable"/> struct.
    /// </summary>
    /// <param name="type">The type of the variable.</param>
    /// <param name="name">The name of the variable.</param>
    public Variable(Type type, string name)
    {
        Type = type;
        Name = name;
    }

    /// <summary>
    /// Gets the type of the variable.
    /// </summary>
    public Type Type { get; }

    /// <summary>
    /// Gets the name of the variable.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the LLVM type of the variable.
    /// </summary>
    public LlvmCompatibleTypes LlvmType => Type.ToLlvmType();

    /// <inheritdoc />
    public static bool operator ==(Variable left, Variable right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Variable left, Variable right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Variable other)
    {
        return Type == other.Type && string.Equals(Name, other.Name, StringComparison.Ordinal);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Variable other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        var hashCode = default(HashCode);
        hashCode.Add(Type);
        hashCode.Add(Name, StringComparer.InvariantCulture);
        return hashCode.ToHashCode();
    }
}