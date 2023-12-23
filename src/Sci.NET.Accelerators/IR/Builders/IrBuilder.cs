// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.Rewriter;
using Sci.NET.Accelerators.IR.Transformers;

namespace Sci.NET.Accelerators.IR.Builders;

/// <summary>
/// Represents an IR builder.
/// </summary>
[PublicAPI]
public static class IrBuilder
{
    private static readonly IReadOnlyCollection<ITransformer> InstructionTransformers = new List<ITransformer> { new CallTransformer(), new ComparisonTransformer() };

    /// <summary>
    /// Builds the IR from the disassembled method.
    /// </summary>
    /// <param name="method">The disassembled method.</param>
    /// <returns>The IR.</returns>
    public static IReadOnlyList<BasicBlock> Build(DisassembledMethod method)
    {
        var cfg = ControlFlowGraph.Create(method.Instructions);
        var executor = new SymbolicExecutor(cfg, method);
        var ssaMethod = executor.Execute();

        _ = ssaMethod.ToString();

        foreach (var transformer in InstructionTransformers)
        {
            transformer.Transform(cfg, method);
        }

        return BasicBlockBuilder.CreateBasicBlocks(cfg);
    }
}