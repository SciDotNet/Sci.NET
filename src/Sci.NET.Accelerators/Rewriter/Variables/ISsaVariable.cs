// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Rewriter.Variables;

/// <summary>
/// An interface for an operand.
/// </summary>
[PublicAPI]
public interface ISsaVariable
{
    /// <summary>
    /// Gets the operand name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the operand declaration type.
    /// </summary>
    public SsaOperandType DeclarationType { get; }

    /// <summary>
    /// Gets the operand type.
    /// </summary>
    public Type Type { get; }

    /// <summary>
    /// Converts the operand to an IR value.
    /// </summary>
    /// <returns>The IR value.</returns>
    public IrValue ToIrValue()
    {
        return new IrValue { Identifier = Name, Type = Type.ToIrType() };
    }
}