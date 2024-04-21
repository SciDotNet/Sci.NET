// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.SymbolStore;
using System.Reflection;
using Sci.NET.Accelerators.Disassembly.Pdb;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a MSIL method metadata.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public class MsilMethodMetadata
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MsilMethodMetadata"/> class.
    /// </summary>
    /// <param name="methodBase">The method base.</param>
    /// <exception cref="InvalidOperationException">Method body is null.</exception>
    [SetsRequiredMembers]
    public MsilMethodMetadata(MethodBase methodBase)
    {
        var debugInfoSuccess = DebugInformationManager.TryLoadMethodDebugInformation(methodBase, out var methodDebugInfo);

        MethodDebugInfo = methodDebugInfo;

        if (!debugInfoSuccess)
        {
            throw new InvalidOperationException("Debug information not loaded.");
        }

        MethodBase = methodBase;
        MethodBody = methodBase.GetMethodBody() ?? throw new InvalidOperationException("Method body is null.");
        ReturnType = methodBase is MethodInfo methodInfo ? methodInfo.ReturnType : typeof(void);
        MaxStack = MethodBody.MaxStackSize;
        CodeSize = MethodBody.GetILAsByteArray()?.Length ?? throw new InvalidOperationException("Method body is null.");
        LocalVariablesSignatureToken = new SymbolToken(MethodBody.LocalSignatureMetadataToken);
        InitLocals = MethodBody.InitLocals;
        Parameters = methodBase.GetParameters().ToArray();
        TypeGenericArguments = methodBase.DeclaringType?.GetGenericArguments().ToArray() ?? Array.Empty<Type>();
        MethodGenericArguments = methodBase.GetGenericArguments().ToArray();
        Module = methodBase.Module;

        var variablesBuilder = ImmutableArray.CreateBuilder<LocalVariable>();

        foreach (var localVariable in MethodBody.LocalVariables)
        {
            var pdbVariable = MethodDebugInfo.LocalVariables.FirstOrDefault(x => x.Index == localVariable.LocalIndex);

            variablesBuilder.Add(
                pdbVariable.Name is not null
                    ? new LocalVariable { Index = localVariable.LocalIndex, Name = pdbVariable.Name, Type = localVariable.LocalType }
                    : new LocalVariable { Index = localVariable.LocalIndex, Name = $"loc_{localVariable.LocalIndex}", Type = localVariable.LocalType });
        }

        Variables = variablesBuilder.ToImmutable();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MsilMethodMetadata"/> class.
    /// </summary>
    public MsilMethodMetadata()
    {
    }

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
    public required SymbolToken LocalVariablesSignatureToken { get; init; }

    /// <summary>
    /// Gets a value indicating whether the method body is init locals.
    /// </summary>
    public required bool InitLocals { get; init; }

    /// <summary>
    /// Gets the parameters.
    /// </summary>
    public required ParameterInfo[] Parameters { get; init; }

    /// <summary>
    /// Gets the local variables.
    /// </summary>
    public required ImmutableArray<LocalVariable> Variables { get; init; }

    /// <summary>
    /// Gets the type generic arguments.
    /// </summary>
    public required Type[] TypeGenericArguments { get; init; }

    /// <summary>
    /// Gets the method generic arguments.
    /// </summary>
    public required Type[] MethodGenericArguments { get; init; }

    /// <summary>
    /// Gets the reflected method base.
    /// </summary>
    public required MethodBase MethodBase { get; init; }

    /// <summary>
    /// Gets the method body.
    /// </summary>
    public required MethodBody MethodBody { get; init; }

    /// <summary>
    /// Gets the return type.
    /// </summary>
    public required Type ReturnType { get; init; }

    /// <summary>
    /// Gets the module.
    /// </summary>
    public required Module Module { get; init; }

    /// <summary>
    /// Gets the method debug information.
    /// </summary>
    public required MethodDebugInfo MethodDebugInfo { get; init; }
}