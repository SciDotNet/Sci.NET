// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a constant operand.
/// </summary>
[PublicAPI]
public class Float4ConstantSsaVariable : IConstantSsaVariable<float>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Float4ConstantSsaVariable"/> class.
    /// </summary>
    /// <param name="name">The name of the operand.</param>
    /// <param name="value">The value of the operand.</param>
    public Float4ConstantSsaVariable(string name, float value)
    {
        Name = name;
        Type = typeof(float);
        Value = value;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.Constant;

    /// <inheritdoc />
    public Type Type { get; }

    /// <inheritdoc />
    public float Value { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Value}, {Type}";
    }
}