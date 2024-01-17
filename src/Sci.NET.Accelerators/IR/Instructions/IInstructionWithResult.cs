// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// Represents an instruction with a result.
/// </summary>
[PublicAPI]
public interface IInstructionWithResult : IInstruction
{
    /// <summary>
    /// Gets the result of the instruction.
    /// </summary>
    public Variable Result { get; }
}