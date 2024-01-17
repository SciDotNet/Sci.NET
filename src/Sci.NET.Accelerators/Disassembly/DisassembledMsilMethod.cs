// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
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
    public required IReadOnlyCollection<ParameterInfo> Parameters { get; init; }

    /// <summary>
    /// Gets the local variables.
    /// </summary>
    public required IReadOnlyCollection<LocalVariableInfo> Variables { get; init; }

    /// <summary>
    /// Gets the instructions.
    /// </summary>
    public required IReadOnlyCollection<MsilInstruction<IMsilOperand>> Instructions { get; init; }

    /// <summary>
    /// Gets the type generic arguments.
    /// </summary>
    public required IReadOnlyList<Type> TypeGenericArguments { get; init; }

    /// <summary>
    /// Gets the method generic arguments.
    /// </summary>
    public required IReadOnlyCollection<Type> MethodGenericArguments { get; init; }

    /// <summary>
    /// Gets the reflected method base.
    /// </summary>
    public required MethodBase ReflectedMethodBase { get; init; }

    /// <summary>
    /// Gets the return type.
    /// </summary>
    public required Type ReturnType { get; init; }
}