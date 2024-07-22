// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Rewriter.Variables;

namespace Sci.NET.Accelerators.Rewriter;

internal class IrGeneratingContext
{
    public IrGeneratingContext(DisassembledMsilMethod method)
    {
        DisassembledMethod = method;
        NameGenerator = new VariableNameGenerator();
        LocalVariableSsaVariables = new LocalVariableSsaVariable[method.Metadata.Variables.Length];
        Stack = new Stack<ISsaVariable>();

        for (var index = 0; index < method.Metadata.Variables.Length; index++)
        {
            LocalVariableSsaVariables[index] = new LocalVariableSsaVariable(
                index,
                NameGenerator.NextLocalName(),
                method.Metadata.Variables[index].Type);
        }

        ArgumentSsaVariables = new ParameterSsaVariable[method.Metadata.Parameters.Length];

        for (var index = 0; index < method.Metadata.Parameters.Length; index++)
        {
            ArgumentSsaVariables[index] = new ParameterSsaVariable(
                index,
                $"arg_{method.Metadata.Parameters[index].Name}",
                method.Metadata.Parameters[index].ParameterType);
        }
    }

    public DisassembledMsilMethod DisassembledMethod { get; }

    public LocalVariableSsaVariable[] LocalVariableSsaVariables { get; }

    public ParameterSsaVariable[] ArgumentSsaVariables { get; }

    public VariableNameGenerator NameGenerator { get; }

    public Stack<ISsaVariable> Stack { get; }
}