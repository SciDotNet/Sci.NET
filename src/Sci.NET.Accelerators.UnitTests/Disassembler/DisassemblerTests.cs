// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Attributes;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Builders;
using Sci.NET.Accelerators.IR.Rewriter;
using Sci.NET.Accelerators.Runtime;

namespace Sci.NET.Accelerators.UnitTests.Disassembler;

public class DisassemblerTests
{
    [Fact]
    public void Test1()
    {
        var methodBase = typeof(DisassemblerTests).GetMethod(nameof(AddKernel));
        var disassembler = new Disassembly.Disassembler(methodBase!);
        var method = disassembler.Disassemble();

        var ir = IrBuilder.Build(method);

        var i = 0;
        var irMethod = new SsaMethod(
            method.Parameters.Select(x => new Parameter(x.ParameterType, i++)).ToList(),
            method.Variables.Select(x => new LocalVariable(x.LocalType)).ToList(),
            ir.ToList(),
            method.ReturnType,
            method.ReflectedMethodBase.Name);

        var str = irMethod.ToString();

        str.Should().NotBeEmpty();
    }

    [Kernel]
#pragma warning disable xUnit1013
    public static unsafe void AddKernel(float* left, float* right, float* result, long length)
#pragma warning restore xUnit1013
    {
        if (ParallelThread.ThreadIdx.X < length)
        {
            result[ParallelThread.ThreadIdx.X] = left[ParallelThread.ThreadIdx.X] + right[ParallelThread.ThreadIdx.X];
        }
    }
}