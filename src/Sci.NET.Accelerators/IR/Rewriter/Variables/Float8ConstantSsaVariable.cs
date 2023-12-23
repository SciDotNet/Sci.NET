// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public class Float8ConstantSsaVariable : IConstantSsaVariable<double>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Float8ConstantSsaVariable"/> class.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Float8ConstantSsaVariable(string name, double value)
    {
        Name = name;
        Type = typeof(double);
        Value = value;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Constant;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public double Value { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Value}, {Type}";
    }
}