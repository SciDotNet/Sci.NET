﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Text;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

internal class RemoveUnusedInstructions : IIrTransform
{
    public void Transform(BasicBlock block, ICollection<BasicBlock> allBlocks)
    {
        var usedValues = new HashSet<IrValue>();
        var removedCount = 0;

        foreach (var instruction in allBlocks.SelectMany(x => x.Instructions))
        {
            foreach (var operand in instruction.Operands)
            {
                _ = usedValues.Add(operand);
            }
        }

        for (var index = 0; index < block.Instructions.Count; index++)
        {
            var instruction = block.Instructions[index];
            if (instruction is not IValueYieldingInstruction yieldingInstruction)
            {
                continue;
            }

            if (!usedValues.Contains(yieldingInstruction.Result) && !yieldingInstruction.Result.Type.Equals(IrType.Void))
            {
                block.RemoveInstruction(index);

                var sb = new StringBuilder();
                _ = yieldingInstruction.WriteToIrString(sb);
                Debug.WriteLine($"Removed instruction: {sb}");

                removedCount++;
            }
        }

        Debug.WriteLine(removedCount);
    }
}