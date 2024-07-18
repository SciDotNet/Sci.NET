// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Pdb;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
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

    /// <summary>
    /// Gets the IR and MSIL string representation of the method.
    /// </summary>
    /// <returns>The IR and MSIL string representation of the method.</returns>
    public string GetIrAndMsilString()
    {
        var headerBuilder = new StringBuilder();
        var lines = new List<(string IR, string MSIL, string CS)>();
        var builder = new StringBuilder();
        _ = headerBuilder.Append("method ").Append(Metadata.MethodBase.Name).Append('(');
        foreach (var parameter in Parameters)
        {
            _ = headerBuilder.Append(parameter.Type).Append(' ').Append(parameter.Name).Append(", ");
        }

        if (Parameters.Count > 0)
        {
            headerBuilder.Length -= 2;
        }

        _ = headerBuilder.Append(") : ").Append(ReturnType).AppendLine(" {");
        foreach (var local in Locals)
        {
            _ = headerBuilder.Append("    ").Append(local.Type).Append(' ').Append(local.Name).AppendLine(";");
        }

        foreach (var basicBlock in BasicBlocks)
        {
            lines.Add((basicBlock.Name + ":", string.Empty, string.Empty));

            foreach (var instruction in basicBlock.Instructions)
            {
                if (instruction is NopInstruction)
                {
                    continue;
                }

                var instructionBuilder = new StringBuilder();
                _ = instruction.WriteToIrString(instructionBuilder);

                var cs = string.Empty;
                var sequencePoint = instruction.MsilInstruction?.SequencePoint;
                var sourceFile = sequencePoint?.DocumentName;

                if (sequencePoint is not null && sourceFile is not null && File.Exists(instruction.MsilInstruction?.SequencePoint?.DocumentName))
                {
                    cs = HandleAddSequencePoint(sequencePoint, sourceFile);
                }

                lines.Add(("    " + instructionBuilder, instruction.MsilInstruction.ToString()!, cs));
            }
        }

        const int paddingSize = 4;
        var longestIrLine = lines.Max(x => x.IR.Length) + paddingSize;
        var longestMsilLine = lines.Max(x => x.MSIL.Length) + paddingSize;

        foreach (var (ir, msil, cs) in lines)
        {
            _ = builder.Append("|".PadRight(paddingSize)).Append(msil.PadRight(longestMsilLine)).Append("|".PadRight(paddingSize)).Append(ir.PadRight(longestIrLine)).Append("|".PadRight(paddingSize)).AppendLine(cs);
        }

        _ = builder.AppendLine("}");
        return builder.ToString();
    }

    private static string HandleAddSequencePoint([DisallowNull] PdbSequencePoint? sequencePoint, string sourceFile)
    {
        string cs;
        try
        {
            if (sequencePoint.Value.IsHidden)
            {
                cs = "#line hidden";
            }
            else
            {
                var source = File.ReadAllLines(sourceFile);
                var line = source[sequencePoint.Value.StartLine - 1].Trim();
                cs = $"{line.Replace("\n", " ", StringComparison.Ordinal)}";
            }
        }
        catch (Exception)
        {
            cs = "// PDB Error: Could not read source file.";
        }

        return cs;
    }
}