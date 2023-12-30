// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.CompilerServices;
using Sci.NET.Accelerators.Attributes;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.Builders;
using Sci.NET.Accelerators.Metadata;
using Sci.NET.Accelerators.UnitTests.Extensions;

namespace Sci.NET.Accelerators.UnitTests.Disassembly;

public class DisassemblerTests
{
    [Fact]
    public void Disassembler_Constructor_ThrowsOnNullMethodFloat()
    {
        var method = typeof(DisassemblerTests).GetMethod(nameof(TestMethodFloatExplicit), BindingFlags.NonPublic | BindingFlags.Static);
        RuntimeHelpers.PrepareMethod(method?.MethodHandle ?? throw new InvalidOperationException("Method handle is null."));

        var disassembler = new Disassembler(method);
        var disassembledMethod = disassembler.Disassemble();

        var irBuilder = IrBuilder.Build(disassembledMethod);

        var str = irBuilder.ConvertToString();
    }

    [Kernel]
    [SuppressMessage("ReSharper", "UnusedMember.Local", Justification = "Test method.")]
    private static unsafe void TestMethodFloatExplicit(
        float* left,
        float* right,
        float* result,
        int m)
    {
        for (var i = 0; i < m; i++)
        {
            LoopMetadata.BeginVectorizedLoop();
            result[i] = left[i] + right[i];
            LoopMetadata.BeginVectorizedLoop();
        }
    }
}