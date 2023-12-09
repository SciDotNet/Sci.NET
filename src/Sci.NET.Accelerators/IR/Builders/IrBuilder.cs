// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.PatternRecognition;
using Sci.NET.Accelerators.IR.Transformers;

namespace Sci.NET.Accelerators.IR.Builders;

/// <summary>
/// Represents an IR builder.
/// </summary>
[PublicAPI]
public static class IrBuilder
{
    private static readonly IReadOnlyCollection<ITransformer> InstructionTransformers = new List<ITransformer>
    {
        new MemoryBlockTransformer()
    };

    /// <summary>
    /// Builds the IR from the disassembled method.
    /// </summary>
    /// <param name="method">The disassembled method.</param>
    /// <returns>The IR.</returns>
    public static IReadOnlyList<BasicBlock> Build(DisassembledMethod method)
    {
        foreach (var transformer in InstructionTransformers)
        {
            transformer.Transform(method);
        }

        return BasicBlockBuilder.CreateBasicBlocks(method.Instructions);
    }
}