// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public class Int8ConstantSsaVariable : IConstantSsaVariable<long>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Int8ConstantSsaVariable"/> class.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Int8ConstantSsaVariable(string name, long value)
    {
        Name = name;
        Value = value;
        Type = typeof(long);
    }

    /// <inheritdoc />
    public long Value { get; }

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