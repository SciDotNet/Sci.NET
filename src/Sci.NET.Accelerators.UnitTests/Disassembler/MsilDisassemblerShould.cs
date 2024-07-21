// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.UnitTests.Disassembler;

public class MsilDisassemblerShould
{
    [Fact]
    public void DisassembleMethod_GivenExample1()
    {
        // Arrange
        var method = GetType().GetMethod(nameof(MultiplyAdd100), BindingFlags.Static | BindingFlags.NonPublic) ?? throw new InvalidOperationException();
        var disassembler = new MsilDisassembler(new MsilMethodMetadata(method));

        // Act
        var disassembledMethod = disassembler.Disassemble();

        // Assert
        Assert.NotNull(disassembledMethod);
    }

    private static int MultiplyAdd100(int left, int right)
    {
        var result = left * right;

        return result + 100;
    }
}