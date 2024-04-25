// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions.Terminators;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

internal class DeadBlockRemover : IBlockTransform
{
    public void Transform(ICollection<BasicBlock> allBlocks)
    {
        var deadBlocks = new HashSet<BasicBlock>();
        foreach (var block in allBlocks)
        {
            if (block.Instructions.Count == 0)
            {
                _ = deadBlocks.Add(block);
            }
        }

        foreach (var block in allBlocks)
        {
            foreach (var instruction in block.Instructions)
            {
                if (instruction is IBranchInstruction branchInstruction && deadBlocks.Contains(branchInstruction.Target))
                {
                    _ = deadBlocks.Add(block);
                }
            }
        }

        foreach (var block in deadBlocks)
        {
            _ = allBlocks.Remove(block);
        }
    }
}