// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// An instruction in the intermediate representation.
/// </summary>
[PublicAPI]
public interface IInstruction : IIrWritable
{
    /// <summary>
    /// Gets the instruction name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the corresponding MSIL instruction (if any).
    /// </summary>
    public MsilInstruction<IMsilOperand>? MsilInstruction { get; }
}