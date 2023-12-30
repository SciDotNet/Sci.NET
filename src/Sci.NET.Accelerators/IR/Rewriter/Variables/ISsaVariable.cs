// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

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
}