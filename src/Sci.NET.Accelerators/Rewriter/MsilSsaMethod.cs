// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.Rewriter.Variables;

namespace Sci.NET.Accelerators.Rewriter;

/// <summary>
/// Represents a method in MSIL (Microsoft Intermediate Language) with Static Single Assignment (SSA) form.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class MsilSsaMethod
{
    /// <summary>
    /// Gets the collection of basic blocks.
    /// </summary>
    public required ICollection<BasicBlock> BasicBlocks { get; init; }

    /// <summary>
    /// Gets the collection of parameters.
    /// </summary>
    public required ICollection<ParameterSsaVariable> Parameters { get; init; }

    /// <summary>
    /// Gets the collection of local variables.
    /// </summary>
    public required ICollection<LocalVariableSsaVariable> Locals { get; init; }

    /// <summary>
    /// Gets the return type.
    /// </summary>
    public required Type ReturnType { get; init; }

    /// <summary>
    /// Gets the metadata of the method.
    /// </summary>
    public required MsilMethodMetadata Metadata { get; init; }

    /// <inheritdoc />
    public override string ToString()
    {
        var builder = new StringBuilder();
        _ = builder.Append("method ").Append(Metadata.MethodBase.Name).Append('(');
        foreach (var parameter in Parameters)
        {
            _ = builder.Append(parameter.Type).Append(' ').Append(parameter.Name).Append(", ");
        }

        if (Parameters.Count > 0)
        {
            builder.Length -= 2;
        }

        _ = builder.Append(") : ").Append(ReturnType).AppendLine(" {");
        foreach (var local in Locals)
        {
            _ = builder.Append("    ").Append(local.Type).Append(' ').Append(local.Name).AppendLine(";");
        }

        foreach (var basicBlock in BasicBlocks)
        {
            _ = builder.AppendLine(basicBlock.ToString());
        }

        _ = builder.AppendLine("}");
        return builder.ToString();
    }
}