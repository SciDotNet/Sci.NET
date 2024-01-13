// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.UnitTests.Assertions;

public class InstructionAssertions : ReferenceTypeAssertions<Instruction<IOperand>, InstructionAssertions>
{
    public InstructionAssertions(Instruction<IOperand> subject)
        : base(subject)
    {
    }

    protected override string Identifier => "instruction";

    public AndConstraint<InstructionAssertions> HaveOpCode(OpCodeTypes opCode)
    {
        Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.IlOpCode)
            .ForCondition(instructionOpCode => instructionOpCode == opCode)
            .FailWith("Expected instruction to have opcode {0}{reason}, but found {1}.", opCode, Subject.IlOpCode);

        return new AndConstraint<InstructionAssertions>(this);
    }

    public AndConstraint<InstructionAssertions> HavePopBehaviour(PopBehaviour popBehaviour)
    {
        Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.PopBehaviour)
            .ForCondition(instructionPopBehaviour => instructionPopBehaviour == popBehaviour)
            .FailWith("Expected instruction to have pop behaviour {0}{reason}, but found {1}.", popBehaviour, Subject.PopBehaviour);

        return new AndConstraint<InstructionAssertions>(this);
    }

    public AndConstraint<InstructionAssertions> HavePushBehaviour(PushBehaviour pushBehaviour)
    {
        Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.PushBehaviour)
            .ForCondition(instructionPushBehaviour => instructionPushBehaviour == pushBehaviour)
            .FailWith("Expected instruction to have push behaviour {0}{reason}, but found {1}.", pushBehaviour, Subject.PushBehaviour);

        return new AndConstraint<InstructionAssertions>(this);
    }

    public AndConstraint<InstructionAssertions> HaveFlowControl(FlowControl flowControl)
    {
        Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.FlowControl)
            .ForCondition(instructionFlowControl => instructionFlowControl == flowControl)
            .FailWith("Expected instruction to have flow control {0}{reason}, but found {1}.", flowControl, Subject.FlowControl);

        return new AndConstraint<InstructionAssertions>(this);
    }

    public AndConstraint<InstructionAssertions> HaveOffset(int offset)
    {
        Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.Offset)
            .ForCondition(instructionOffset => instructionOffset == offset)
            .FailWith("Expected instruction to have offset {0}{reason}, but found {1}.", offset, Subject.Offset);

        return new AndConstraint<InstructionAssertions>(this);
    }

    public AndConstraint<InstructionAssertions> MatchOperand(Func<IOperand, bool> operand)
    {
        Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.Operand)
            .ForCondition(_ => operand(Subject.Operand))
            .FailWith("Expected instruction to have operand {0}{reason}, but found {1}.", operand, Subject.Operand);

        return new AndConstraint<InstructionAssertions>(this);
    }
}