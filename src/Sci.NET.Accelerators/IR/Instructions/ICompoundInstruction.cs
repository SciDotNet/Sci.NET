// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// Represents a compound instruction.
/// </summary>
[PublicAPI]
public interface ICompoundInstruction : IInstruction
{
    /// <summary>
    /// Gets the instructions that make up the compound instruction.
    /// </summary>
    public ICollection<IInstruction> Instructions { get; }
}