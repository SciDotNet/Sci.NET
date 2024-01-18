// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.SymbolStore;
using System.Reflection;

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
        MethodBase = methodBase;
        MethodBody = methodBase.GetMethodBody() ?? throw new InvalidOperationException("Method body is null.");
        ReturnType = methodBase is MethodInfo methodInfo ? methodInfo.ReturnType : typeof(void);
        MaxStack = MethodBody.MaxStackSize;
        CodeSize = MethodBody.GetILAsByteArray()?.Length ?? throw new InvalidOperationException("Method body is null.");
        LocalVariablesSignatureToken = new SymbolToken(MethodBody.LocalSignatureMetadataToken);
        InitLocals = MethodBody.InitLocals;
        Parameters = methodBase.GetParameters().ToArray();
        Variables = MethodBody.LocalVariables.ToArray();
        TypeGenericArguments = methodBase.DeclaringType?.GetGenericArguments().ToArray() ?? Array.Empty<Type>();
        MethodGenericArguments = methodBase.GetGenericArguments().ToArray();
        Module = methodBase.Module;
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
    public required LocalVariableInfo[] Variables { get; init; }

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
}