// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Rewriter;

namespace Sci.NET.Accelerators.UnitTests.Rewriter;

public class CfgBuilderShould
{
    [Fact]
    public void SomeTest()
    {
        var method = GetType().GetMethod(nameof(Add), BindingFlags.Static | BindingFlags.NonPublic) ?? throw new InvalidOperationException();
        var disassembledMethod = new MsilDisassembler(new MsilMethodMetadata(method)).Disassemble();

        // Act
        var cfg = CfgBuilder.Build(disassembledMethod);

        // Assert
        Assert.NotNull(cfg);
    }

    private static int Add(int left, int right)
    {
        var result = left * right;

        if (result > 100)
        {
            return result;
        }

        return result + 100;
    }
}