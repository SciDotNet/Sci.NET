// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.UnitTests.Disassembly;

public class DisassemblerTests
{
    [Fact]
    public void Disassembler_Constructor_ThrowsOnNullMethod()
    {
        var targetMethod = typeof(DisassemblerTests).GetMethod(nameof(TestMethod), BindingFlags.NonPublic | BindingFlags.Static);

        var disassembler = new Disassembler(targetMethod!);

        var instructions = disassembler.Disassemble();

        var instructionStrings = instructions.ToString();
    }

    [SuppressMessage("ReSharper", "UnusedMember.Local", Justification = "Test method.")]
    private static void TestMethod(int left, int right)
    {
        var x = left;
        var y = right;
        var z = x + y;

#pragma warning disable CA1303
        if (z > 0)
        {
            Console.WriteLine("Z is greater than 0.");
        }
        else
        {
            Console.WriteLine("Z is less than or equal to 0.");
        }
#pragma warning restore CA1303
    }
}