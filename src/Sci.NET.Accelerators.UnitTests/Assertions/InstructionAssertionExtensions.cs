// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.UnitTests.Assertions;

public static class InstructionAssertionExtensions
{
    public static InstructionAssertions Should(this Instruction<IOperand> instruction)
    {
        return new InstructionAssertions(instruction);
    }
}