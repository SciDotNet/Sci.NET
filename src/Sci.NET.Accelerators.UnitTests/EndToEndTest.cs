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
        var metadata = typeof(EndToEndTest).GetMethod(nameof(MatrixMultiply), BindingFlags.Static | BindingFlags.NonPublic) ??
                       throw new InvalidOperationException();

        var info = new MsilMethodMetadata(metadata);
        var disassembler = new MsilDisassembler(info);
        var disassembly = disassembler.Disassemble();
        var ssaConverter = new MsilToIrTranslator(disassembly);
        var cfg = CfgBuilder.Build(disassembly);
        var ir = ssaConverter.Transform(cfg);

        // I need to write some real assertions here, but for now, just check that the IR is not null
        Assert.NotNull(ir);
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
        var cfg = CfgBuilder.Build(disassembly);
        var ir = ssaConverter.Transform(cfg);

        // I need to write some real assertions here, but for now, just check that the IR is not null
        Assert.NotNull(ir);
    }

    private static void MatrixMultiply(
        IMemoryBlock<float> left,
        IMemoryBlock<float> right,
        IMemoryBlock<float> result,
        long m,
        long n,
        long k)
    {
        var row = (Kernel.BlockIdx.Y * Kernel.BlockDim.Y) + Kernel.ThreadIdx.Y;
        var col = (Kernel.BlockIdx.X * Kernel.BlockDim.X) + Kernel.ThreadIdx.X;
        var sum = 0.0f;

        if (row < m && col < n)
        {
            for (var i = 0; i < k; i++)
            {
                sum += left[(row * k) + i] * right[(i * n) + col];
            }
        }

        Kernel.SyncThreads();

        result[(row * n) + col] = sum;
    }

    private static int AddOther(int left, int right)
    {
        return left + right;
    }
}