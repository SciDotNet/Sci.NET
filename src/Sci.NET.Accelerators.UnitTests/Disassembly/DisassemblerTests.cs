// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.Builders;
using Sci.NET.Accelerators.Metadata;
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

        var basicBlocks = BasicBlockBuilder.CreateBasicBlocks(disassembledMethod.Instructions.ToList());
    }

    [Fact]
    public void Disassembler_Constructor_ThrowsOnNullMethodFloat()
    {
        var method = typeof(DisassemblerTests).GetMethod(nameof(TestMethodFloatExplicit), BindingFlags.NonPublic | BindingFlags.Static);
        RuntimeHelpers.PrepareMethod(method.MethodHandle);

        var disassembler = new Disassembler(method);
        var disassembledMethod = disassembler.Disassemble();

        var builder = IrBuilder.Build(disassembledMethod);
    }

    [SuppressMessage("ReSharper", "UnusedMember.Local", Justification = "Test method.")]
    private static void TestMethod<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        LoopMetadata.StartBoundaryCondition();

        for (var i = 0L; i < left.Length; i++)
        {
            LoopMetadata.EndBoundaryCondition();
            LoopMetadata.Start();
            LoopMetadata.StartBoundaryCondition();

            for (var j = 0L; j < right.Length; j++)
            {
                LoopMetadata.EndBoundaryCondition();
                LoopMetadata.Start();

                result[i] = left[i] + right[j];

                LoopMetadata.End();
            }

            LoopMetadata.End();
        }
    }

    [SuppressMessage("ReSharper", "UnusedMember.Local", Justification = "Test method.")]
    private static void TestMethodFloatExplicit(IMemoryBlock<float> left, IMemoryBlock<float> right, IMemoryBlock<float> result)
    {
        LoopMetadata.StartBoundaryCondition();

        for (var i = 0L; i < left.Length; i++)
        {
            LoopMetadata.EndBoundaryCondition();
            LoopMetadata.Start();
            LoopMetadata.StartBoundaryCondition();

            for (var j = 0L; j < right.Length; j++)
            {
                LoopMetadata.EndBoundaryCondition();
                LoopMetadata.Start();

                result[i] = left[i] + right[j];

                LoopMetadata.End();
            }

            LoopMetadata.End();
        }
    }
}