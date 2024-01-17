// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.Disassembly.Instructions;

/// <summary>
/// Represents a comparison node.
/// </summary>
[PublicAPI]
public class ComparisonNode : IMsilControlFlowGraphNode
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ComparisonNode"/> class.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="comparison">The comparison.</param>
    /// <param name="replacingNodes">The nodes to replace.</param>
    public ComparisonNode(
        IMsilControlFlowGraphNode leftOperand,
        IMsilControlFlowGraphNode rightOperand,
        IMsilControlFlowGraphNode comparison,
        IEnumerable<IMsilControlFlowGraphNode> replacingNodes)
    {
        var controlFlowGraphNodes = replacingNodes.ToList();

        LeftOperand = leftOperand;
        RightOperand = rightOperand;
        Comparison = comparison;
        Instruction = new Instruction<IOperand>
        {
            Operand = default(NoOperand),
            IlOpCode = OpCodeTypes.Nop,
            Offset = controlFlowGraphNodes[0].Instruction.Offset,
            Size = controlFlowGraphNodes.Sum(x => x.Instruction.Size),
            Name = "Comparison",
            PopBehaviour = PopBehaviour.Pop1_pop1,
            PushBehaviour = PushBehaviour.Push1,
            FlowControl = FlowControl.Next,
            OperandType = OperandType.InlineNone,
            IsMacro = false,
            Index = controlFlowGraphNodes[0].Instruction.Index
        };
        NextInstructions = controlFlowGraphNodes.SelectMany(x => x.NextInstructions).ToList();
        IsLeader = controlFlowGraphNodes.Exists(x => x.IsLeader);
        IsTerminator = controlFlowGraphNodes.Exists(x => x.IsTerminator);
    }

    /// <summary>
    /// Gets the right operand.
    /// </summary>
    public IMsilControlFlowGraphNode LeftOperand { get; }

    /// <summary>
    /// Gets the left operand.
    /// </summary>
    public IMsilControlFlowGraphNode RightOperand { get; }

    /// <summary>
    /// Gets the comparison.
    /// </summary>
    public IMsilControlFlowGraphNode Comparison { get; }

    /// <inheritdoc />
    public Instruction<IOperand> Instruction { get; }

    /// <inheritdoc />
    public IList<Instruction<IOperand>> NextInstructions { get; }

    /// <inheritdoc />
    public bool IsLeader { get; set; }

    /// <inheritdoc />
    public bool IsTerminator { get; set; }
}