﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

/// <summary>
/// An interface for an instruction transform.
/// </summary>
[PublicAPI]
public interface IIrTransform
{
    /// <summary>
    /// Transforms the instructions in a basic block.
    /// </summary>
    /// <param name="block">The basic block to transform.</param>
    /// <param name="allBlocks">All basic blocks in the function.</param>
    public void Transform(BasicBlock block, ICollection<BasicBlock> allBlocks);
}