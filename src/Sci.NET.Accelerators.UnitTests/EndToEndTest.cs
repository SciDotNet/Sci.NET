// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Rewriter;
using Sci.NET.Common.Memory;

namespace Sci.NET.Accelerators.UnitTests;

public class EndToEndTest
{
    [Fact]
    public void Test()
    {
        var metadata = typeof(EndToEndTest).GetMethod(nameof(Add), BindingFlags.Static | BindingFlags.NonPublic) ??
                       throw new InvalidOperationException();

        var info = new MsilMethodMetadata(metadata);
        var disassembler = new MsilDisassembler(info);
        var disassembly = disassembler.Disassemble();
        var ssaConverter = new MsilToIrTranslator(disassembly);
        var cfg = new CfgBuilder(disassembly).Build();
        var ir = ssaConverter.Transform(cfg);

        var irString = ir.GetIrAndMsilString();

        Assert.NotNull(disassembly);
    }

    [Fact]
    public void Test2()
    {
        var metadata = typeof(EndToEndTest).GetMethod(nameof(AddOther), BindingFlags.Static | BindingFlags.NonPublic) ??
                       throw new InvalidOperationException();

        var info = new MsilMethodMetadata(metadata);
        var disassembler = new MsilDisassembler(info);
        var disassembly = disassembler.Disassemble();
        var ssaConverter = new MsilToIrTranslator(disassembly);
        var cfg = new CfgBuilder(disassembly).Build();
        var ir = ssaConverter.Transform(cfg);

        var irString = ir.GetIrAndMsilString();

        Assert.NotNull(ir);
    }

    private static void Add(
        IMemoryBlock<float> left,
        IMemoryBlock<float> right,
        IMemoryBlock<float> result,
        long m,
        long n,
        long k)
    {
        // Basic block 1
        var row = (Kernel.BlockIdx.Y * Kernel.BlockDim.Y) + Kernel.ThreadIdx.Y;
        var col = (Kernel.BlockIdx.X * Kernel.BlockDim.X) + Kernel.ThreadIdx.X;
        var sum = 0.0f;

        if (row < m && col < n)
        {
            // Basic block 2
            for (var i = 0; i < k; i++)
            {
                // Basic block 3
                sum += left[(row * k) + i] * right[(i * n) + col];
            }
        }

        // Basic block 4
        Kernel.SyncThreads();

        result[(row * n) + col] = sum;
    }

    private static int AddOther(int left, int right)
    {
        return left + right;
    }
}