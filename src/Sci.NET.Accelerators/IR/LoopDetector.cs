// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a loop detector.
/// </summary>
[PublicAPI]
public static class LoopDetector
{
    /// <summary>
    /// Detects the loops.
    /// </summary>
    /// <param name="basicBlocks">The basic blocks.</param>
    public static void Detect(BasicBlockCollection basicBlocks)
    {
        foreach (var block in basicBlocks)
        {
            if (block.IsLoop)
            {
                Debug.Write("Loop: ");

                if (block.Instructions[^1].IsConditionalBranch)
                {
                    var operands = block.Instructions[^1].Instruction.Operands;

                    Debug.WriteLine("Brtrue");
                    Debug.WriteLine($"    {operands}");
                }
            }
        }
    }
}