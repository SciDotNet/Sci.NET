// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Nodes;

/// <summary>
/// Represents a method call node.
/// </summary>
[PublicAPI]
public class MethodCallNode : IControlFlowGraphNode
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MethodCallNode"/> class.
    /// </summary>
    /// <param name="method">The method being called.</param>
    /// <param name="replacingNodes">The nodes to replace.</param>
    public MethodCallNode(MethodBase method, IEnumerable<IControlFlowGraphNode> replacingNodes)
    {
        var controlFlowGraphNodes = replacingNodes.ToList();

        TargetMethod = method;
        Instruction = new Instruction<IOperand>
        {
            Offset = controlFlowGraphNodes[0].Instruction.Offset,
            Size = controlFlowGraphNodes.Sum(x => x.Instruction.Size),
            Operand = default(NoOperand),
            OpCode = OpCodeTypes.Nop,
            Name = "MethodCall",
            PopBehaviour = PopBehaviour.None,
            PushBehaviour = PushBehaviour.Push0,
            FlowControl = FlowControl.Call,
            OperandType = OperandType.InlineNone,
            IsMacro = false,
            Index = controlFlowGraphNodes[0].Instruction.Index
        };
        NextInstructions = controlFlowGraphNodes.SelectMany(x => x.NextInstructions).ToList();
        IsLeader = controlFlowGraphNodes.Exists(x => x.IsLeader);
        IsTerminator = controlFlowGraphNodes.Exists(x => x.IsTerminator);
    }

    /// <summary>
    /// Gets the target method.
    /// </summary>
    public MethodBase TargetMethod { get; }

    /// <inheritdoc />
    public Instruction<IOperand> Instruction { get; }

    /// <inheritdoc />
    public IList<Instruction<IOperand>> NextInstructions { get; }

    /// <inheritdoc />
    public bool IsLeader { get; set; }

    /// <inheritdoc />
    public bool IsTerminator { get; set; }
}