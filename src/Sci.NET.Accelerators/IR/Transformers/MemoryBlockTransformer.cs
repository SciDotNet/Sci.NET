// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Instructions.Operands;
using Sci.NET.Accelerators.IR.PatternRecognition;
using Sci.NET.Common.Memory;

namespace Sci.NET.Accelerators.IR.Transformers;

/// <summary>
/// Transforms instructions related to memory blocks into pointers and array offset calls.
/// </summary>
[PublicAPI]
public class MemoryBlockTransformer : ITransformer
{
    /// <inheritdoc />
    public void Transform(DisassembledMethod method)
    {
        _ = TransformParameters(method);
        TransformGetElementPointer(method);
    }

    private static void TransformGetElementPointer(DisassembledMethod method)
    {
        foreach (var instruction in method.Instructions)
        {
            var targetMethod = typeof(IMemoryBlock<>)
                .GetMethod("get_Item", BindingFlags.GetProperty | BindingFlags.Instance | BindingFlags.Public);

            if (instruction.Operand is MethodOperand methodOperand && methodOperand.MethodBase == targetMethod)
            {
                _ = methodOperand;
            }
        }
    }

    private static Dictionary<int, int> TransformParameters(DisassembledMethod method)
    {
        var memoryBlockLengths = new Dictionary<int, int>();

        for (var i = 0; i < method.Parameters.Count; i++)
        {
            var type = Type.GetTypeFromHandle(method.Parameters[i].TypeHandle);

            if (type?.Name == typeof(IMemoryBlock<>).Name)
            {
                method.Parameters.Add(typeof(long));

                var newType = typeof(IMemoryBlock<>)
                    .MakeGenericType(type.GenericTypeArguments[0] ?? throw new InvalidOperationException())
                    .GenericTypeArguments[0]
                    .MakePointerType();

                _ = newType;

                method.Parameters[i] = newType;
                memoryBlockLengths.Add(i, method.Parameters.Count - 1);
            }
        }

        return memoryBlockLengths;
    }
}