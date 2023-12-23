// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a temporary operand.
/// </summary>
[PublicAPI]
public class TempSsaVariable : ISsaVariable
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TempSsaVariable"/> class.
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
    public override string ToString()
    {
        return $"{Name}, {Type}";
    }
}