// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a disassembled method body.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public class DisassembledMethod
{
    /// <summary>
    /// Gets the maximum stack size.
    /// </summary>
    public required int MaxStack { get; init; }

    /// <summary>
    /// Gets the size of the code.
    /// </summary>
    public required int CodeSize { get; init; }

    /// <summary>
    /// Gets the local variables signature token.
    /// </summary>
    public required int LocalVariablesSignatureToken { get; init; }

    /// <summary>
    /// Gets a value indicating whether the method body is init locals.
    /// </summary>
    public required bool InitLocals { get; init; }

    /// <summary>
    /// Gets the parameters.
    /// </summary>
    public required IList<ParameterInfo> Parameters { get; init; }

    /// <summary>
    /// Gets the local variables.
    /// </summary>
    public required IList<LocalVariableInfo> Variables { get; init; }

    /// <summary>
    /// Gets the instructions.
    /// </summary>
    public required IList<Instruction<IOperand>> Instructions { get; init; }

    /// <summary>
    /// Gets the type generic arguments.
    /// </summary>
    public required IReadOnlyList<Type> TypeGenericArguments { get; init; }

    /// <summary>
    /// Gets the method generic arguments.
    /// </summary>
    public required IReadOnlyList<Type> MethodGenericArguments { get; init; }

    /// <summary>
    /// Gets the reflected method base.
    /// </summary>
    public required MethodBase ReflectedMethodBase { get; init; }

    /// <summary>
    /// Gets the return type.
    /// </summary>
    public required Type ReturnType { get; init; }

    /// <inheritdoc />
    public override string ToString()
    {
        var builder = new StringBuilder();

#pragma warning disable CA1305, IDE0058
        builder
            .Append("MaxStack: ")
            .Append(MaxStack)
            .AppendLine()
            .Append("CodeSize: ")
            .Append(CodeSize)
            .AppendLine()
            .Append("LocalVariablesSignatureToken: ")
            .Append(LocalVariablesSignatureToken)
            .AppendLine()
            .Append("InitLocals: ")
            .Append(InitLocals)
            .AppendLine()
            .Append("Variables: ")
            .Append(Variables.Count)
            .AppendLine()
            .Append("Instructions: ")
            .Append(Instructions.Count)
            .AppendLine()
            .Append("TypeGenericArguments: ")
            .Append(TypeGenericArguments.Count)
            .AppendLine()
            .Append("MethodGenericArguments: ")
            .Append(MethodGenericArguments.Count)
            .AppendLine()
            .AppendLine();

        foreach (var ins in Instructions)
        {
            builder
                .Append("IL_")
                .Append(ins.Offset.ToString("x4"))
                .Append(' ')
                .Append(ins.Name)
                .Append(' ')
                .Append(ins.Operand)
                .AppendLine();
        }
#pragma warning restore IDE0058, CA1305

        return builder.ToString();
    }
}