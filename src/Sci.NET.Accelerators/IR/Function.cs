// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// A function in the intermediate representation.
/// </summary>
[PublicAPI]
public class Function
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// </summary>
    /// <param name="name">The name of the function.</param>
    /// <param name="parameters">The function parameters.</param>
    /// <param name="returnType">The return type of the function.</param>
    /// <param name="basicBlocks">The basic blocks of the function.</param>
    public Function(string name, ICollection<Parameter> parameters, IrType returnType, ICollection<BasicBlock> basicBlocks)
    {
        Name = name;
        Parameters = parameters;
        ReturnType = returnType;
        BasicBlocks = basicBlocks;
    }

    /// <summary>
    /// Gets the name of the function.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the function parameters.
    /// </summary>
    public ICollection<Parameter> Parameters { get; }

    /// <summary>
    /// Gets the return type of the function.
    /// </summary>
    public IrType ReturnType { get; }

    /// <summary>
    /// Gets the basic blocks of the function.
    /// </summary>
    public ICollection<BasicBlock> BasicBlocks { get; }

    /// <summary>
    /// Gets the function signature.
    /// </summary>
    /// <param name="method">The disassembled method.</param>
    /// <returns>The function name.</returns>
    public static string GetFunctionNameFromMethod(DisassembledMsilMethod method)
    {
        var typeGenericArgs = string.Join('_', method.Metadata.TypeGenericArguments.Select(x => x.Name));
        var methodGenericArgs = string.Join('_', method.Metadata.MethodGenericArguments.Select(x => x.Name));

        return $"{typeGenericArgs}_{method.Metadata.MethodBase.Name}_{methodGenericArgs}";
    }
}