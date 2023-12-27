// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Sci.NET.Accelerators.Attributes;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.Builders;
using Sci.NET.Accelerators.UnitTests.Extensions;
using Sci.NET.Common.Memory;

namespace Sci.NET.Accelerators.UnitTests.Disassembly;

public class DisassemblerTests
{
    [Fact]
    public void Disassembler_Constructor_ThrowsOnNullMethod()
    {
        var method = typeof(DisassemblerTests).GetMethod(nameof(TestMethod), BindingFlags.NonPublic | BindingFlags.Static).MakeGenericMethod(typeof(float));
        RuntimeHelpers.PrepareMethod(method.MethodHandle);

        var disassembler = new Disassembler(method);
        var disassembledMethod = disassembler.Disassemble();

        var irBuilder = IrBuilder.Build(disassembledMethod);
    }

    [Fact]
    public void Disassembler_Constructor_ThrowsOnNullMethodFloat()
    {
        var method = typeof(DisassemblerTests).GetMethod(nameof(TestMethodFloatExplicit), BindingFlags.NonPublic | BindingFlags.Static);
        RuntimeHelpers.PrepareMethod(method.MethodHandle);

        var disassembler = new Disassembler(method);
        var disassembledMethod = disassembler.Disassemble();

        var irBuilder = IrBuilder.Build(disassembledMethod);

        var str = irBuilder.ConvertToString();
    }

    [Kernel]
    [SuppressMessage("ReSharper", "UnusedMember.Local", Justification = "Test method.")]
    private static void TestMethod<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        for (var i = 0L; i < left.Length; i++)
        {
            for (var j = 0L; j < right.Length; j++)
            {
                result[i] = left[i] + right[j];
            }
        }
    }

    [Kernel]
    [SuppressMessage("ReSharper", "UnusedMember.Local", Justification = "Test method.")]
    private static unsafe void TestMethodFloatExplicit(
        float* left,
        float* right,
        float* result,
        int m,
        int n,
        int o)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var sum = 0.0f;

                for (var k = 0; k < o; k++)
                {
                    sum += left[(i * m) + k] *
                           right[(k * k) + j];
                }

                result[(i * n) + j] = sum;
            }
        }
    }
}