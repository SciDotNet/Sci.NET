// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a disassembled MSIL method.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public class DisassembledMsilMethod
{
    /// <summary>
    /// Gets the MSIL method metadata.
    /// </summary>
    public required MsilMethodMetadata Metadata { get; init; }

    /// <summary>
    /// Gets the instructions.
    /// </summary>
    public required IReadOnlyCollection<MsilInstruction<IMsilOperand>> Instructions { get; init; }

    /// <summary>
    /// Gets the name of the method.
    /// </summary>
    public string Name => Metadata.MethodBase.Name;

    /// <inheritdoc />
    public override string ToString()
    {
        var builder = new StringBuilder();

        _ = builder.Append(".method ");

        if (Metadata.MethodBase.IsStatic)
        {
            _ = builder.Append("static ");
        }

        _ = builder.Append(Metadata.ReturnType).AppendLine().Append("  ").Append(Name).Append('(').AppendLine();

        foreach (var parameterInfo in Metadata.Parameters)
        {
            _ = builder.Append("    ").Append(parameterInfo.ParameterType).Append(' ').Append(parameterInfo.Name).AppendLine(", ");
        }

        _ = builder
            .AppendLine("  ) cil managed")
            .AppendLine("  {")
            .Append("    .maxstack  ")
            .Append(Metadata.MaxStack)
            .AppendLine()
            .AppendLine("    .locals init (");

        foreach (var local in Metadata.Variables.OrderBy(x => x.Index))
        {
            _ = builder.Append("      [").Append(local.Index).Append(']').Append(local.Type).Append(' ').Append(local.Name).AppendLine(", ");
        }

        _ = builder
            .AppendLine("    )")
            .AppendLine();

        foreach (var instruction in Instructions)
        {
            _ = builder.Append("    ").AppendLine(instruction.ToString());
        }

        _ = builder.AppendLine("  }");

        return builder.ToString();
    }
}