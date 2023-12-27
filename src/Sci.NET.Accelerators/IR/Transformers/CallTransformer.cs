// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.IR.Nodes;

namespace Sci.NET.Accelerators.IR.Transformers;

/// <summary>
/// Represents a call transformer.
/// </summary>
[PublicAPI]
public class CallTransformer : BaseTransformer
{
    /// <inheritdoc />
    public override void Transform(MsilControlFlowGraph graph, DisassembledMethod context)
    {
        for (var i = 0; i < graph.Nodes.Count; i++)
        {
            var instruction = graph.Nodes[i];

            if (instruction.Instruction.IlOpCode is OpCodeTypes.Call or OpCodeTypes.Callvirt or OpCodeTypes.Calli)
            {
                var (removed, replacement) = ExtractMethodCall(graph, instruction, i);
                var removedNodesArray = removed as IMsilControlFlowGraphNode[] ?? removed.ToArray();

                Reconnect(graph, removedNodesArray, replacement);

                i -= removedNodesArray.Length;
            }
        }
    }

    private static (IEnumerable<IMsilControlFlowGraphNode> Removed, IMsilControlFlowGraphNode Replacement) ExtractMethodCall(MsilControlFlowGraph graph, IMsilControlFlowGraphNode instruction, int i)
    {
        var target = instruction.Instruction.Operand is MethodOperand operand ? operand : throw new InvalidOperationException("The operand is not a method operand.");
        var popCount = target.MethodBase?.GetParameters().Length ?? throw new InvalidOperationException("The method operand does not have a method base.");
        var startIndex = i;

        for (var j = 0; j < popCount;)
        {
            var previousInstruction = graph.Nodes[--startIndex];

            if (IsLoadingArgument(previousInstruction))
            {
                j++;
            }
        }

        var removedNodes = new List<IMsilControlFlowGraphNode>();

        var callNode = new MethodCallNode(
            target.MethodBase,
            removedNodes);

        for (var idx = startIndex + 1; idx < i; idx++)
        {
            removedNodes.Add(graph.Nodes[idx]);
            graph.Nodes.RemoveAt(idx);
        }

        removedNodes.Add(graph.Nodes[i]);
        graph.Nodes.RemoveAt(i);
        graph.Nodes.Insert(startIndex, callNode);

        return (removedNodes, callNode);
    }

    private static bool IsLoadingArgument(IMsilControlFlowGraphNode previousInstruction)
    {
        return previousInstruction.Instruction.PushBehaviour
            is PushBehaviour.Push1
            or PushBehaviour.Push1_push1
            or PushBehaviour.Pushi
            or PushBehaviour.Pushi8
            or PushBehaviour.Pushr4
            or PushBehaviour.Pushi8
            or PushBehaviour.Pushref
            or PushBehaviour.Varpush;
    }
}