﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Instructions.Terminators;

/// <summary>
/// Represents an instruction that branches to a target.
/// </summary>
[PublicAPI]
public interface IConditionalBranchInstruction : IBranchInstruction
{
    /// <summary>
    /// Gets the target of the branch.
    /// </summary>
    public BasicBlock FalseTarget { get; }
}