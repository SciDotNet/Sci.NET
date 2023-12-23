// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a local variable SSA operand.
/// </summary>
[PublicAPI]
public class LocalVariableSsaVariable : ISsaVariable
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LocalVariableSsaVariable"/> class.
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
    public override string ToString()
    {
        return $"{Name}, {Type}";
    }
}