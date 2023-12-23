// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a transformer.
/// </summary>
[PublicAPI]
public abstract class BaseTransformer : ITransformer
{
    /// <inheritdoc />
    public abstract void Transform(ControlFlowGraph graph, DisassembledMethod context);

    /// <inheritdoc />
    public virtual void Reconnect(ControlFlowGraph graph, IList<IControlFlowGraphNode> removed, IControlFlowGraphNode replacement)
    {
        for (var index = 0; index < graph.Nodes.Count; index++)
        {
            var node = graph.Nodes[index];

            for (var i = 0; i < removed.Count; i++)
            {
                var removedNode = removed[i];

                for (var removedIdx = 0; removedIdx < node.NextInstructions.Count; removedIdx++)
                {
                    var nextInstruction = node.NextInstructions[removedIdx];

                    if (nextInstruction == removedNode.Instruction)
                    {
                        _ = node.NextInstructions.Remove(nextInstruction);
                        node.NextInstructions.Add(replacement.Instruction);
                    }
                }
            }
        }
    }
}