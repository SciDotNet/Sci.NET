// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;

namespace Sci.NET.Accelerators.IR.Rewriter;

/// <summary>
/// Represents an SSA method.
/// </summary>
[PublicAPI]
public class SsaMethod
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SsaMethod"/> class.
    /// </summary>
    /// <param name="parameters">The parameters of the method.</param>
    /// <param name="localVariables">The local variables of the method.</param>
    /// <param name="basicBlocks">The basic blocks of the method.</param>
    /// <param name="returnType">The return type of the method.</param>
    /// <param name="name">The name of the method.</param>
    public SsaMethod(ICollection<Parameter> parameters, ICollection<LocalVariable> localVariables, ICollection<BasicBlock> basicBlocks, Type returnType, string name)
    {
        Parameters = parameters;
        LocalVariables = localVariables;
        BasicBlocks = basicBlocks;
        ReturnType = returnType;
        Name = name;
    }

    /// <summary>
    /// Gets the parameters of the method.
    /// </summary>
    public ICollection<Parameter> Parameters { get; }

    /// <summary>
    /// Gets the local variables of the method.
    /// </summary>
    public ICollection<LocalVariable> LocalVariables { get; }

    /// <summary>
    /// Gets the basic blocks of the method.
    /// </summary>
    public ICollection<BasicBlock> BasicBlocks { get; }

    /// <summary>
    /// Gets the return type of the method.
    /// </summary>
    public Type ReturnType { get; }

    /// <summary>
    /// Gets the name of the method.
    /// </summary>
    public string Name { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        var sb = new StringBuilder();

        _ = sb.Append("define ").Append(ReturnType.ToLlvmType().GetCompilerString()).Append(" @\"").Append(Name).Append('"').Append(" (");

        foreach (var parameter in Parameters.Take(Parameters.Count - 1))
        {
            _ = sb.Append(parameter.Type.ToLlvmType().GetCompilerString()).Append(" %").Append(parameter.Index).Append(", ");
        }

        _ = sb.Append(Parameters.Last().Type.ToLlvmType().GetCompilerString()).Append(" %").Append(Parameters.Last().Index).AppendLine(") {");

        foreach (var basicBlock in BasicBlocks)
        {
            _ = sb.AppendLine(basicBlock.ToString());
        }

        _ = sb.AppendLine("}");

        return sb.ToString();
    }
}