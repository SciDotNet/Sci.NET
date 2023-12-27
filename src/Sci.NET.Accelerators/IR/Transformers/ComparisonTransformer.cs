// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Accelerators.IR.Nodes;

namespace Sci.NET.Accelerators.IR.Transformers;

/// <summary>
/// Represents a comparison transformer.
/// </summary>
[PublicAPI]
public class ComparisonTransformer : BaseTransformer
{
    /// <inheritdoc />
    public override void Transform(MsilControlFlowGraph graph, DisassembledMethod context)
    {
        for (var index = 0; index < graph.Nodes.Count; index++)
        {
            var instruction = graph.Nodes[index];

            if (instruction.Instruction.IlOpCode is OpCodeTypes.Clt or OpCodeTypes.Clt_Un or OpCodeTypes.Cgt or OpCodeTypes.Cgt_Un or OpCodeTypes.Ceq)
            {
                var left = graph.Nodes[index - 2];
                var right = graph.Nodes[index - 1];
                var comparison = graph.Nodes[index];
                var removedNodes = new List<IMsilControlFlowGraphNode> { left, right, comparison };

                context.Instructions.RemoveAt(index);
                context.Instructions.RemoveAt(index - 1);
                context.Instructions.RemoveAt(index - 2);

                var comparisonNode = new ComparisonNode(
                    left,
                    right,
                    comparison,
                    removedNodes);

                graph.Nodes.Insert(index - 2, comparisonNode);

                index -= 2;

                Reconnect(graph, removedNodes, comparisonNode);
            }
        }
    }
}