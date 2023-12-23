// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents a void operand used to represent an operation with no result.
/// </summary>
[PublicAPI]
public class VoidSsaVariable : ISsaVariable
{
    /// <inheritdoc />
    public string Name => string.Empty;

    /// <inheritdoc />
    public SsaOperandType DeclarationType => SsaOperandType.None;

    /// <inheritdoc />
    public Type Type => typeof(void);
}