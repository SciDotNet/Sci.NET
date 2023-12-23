// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public class Int4ConstantSsaVariable : IConstantSsaVariable<int>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Int4ConstantSsaVariable"/> class.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Int4ConstantSsaVariable(string name, int value)
    {
        Name = name;
        Value = value;
        Type = typeof(int);
    }

    /// <inheritdoc />
    public int Value { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Constant;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Value}, {Type}";
    }
}