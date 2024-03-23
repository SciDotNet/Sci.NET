// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Metadata;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// Represents the debug information for a method.
/// </summary>
[PublicAPI]
public class MethodDebugInfo
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MethodDebugInfo"/> class.
    /// </summary>
    /// <param name="assemblyDebugInformation">The assembly debug information.</param>
    /// <param name="methodBase">The method base.</param>
    /// <param name="definitionHandle">The method definition handle.</param>
    public MethodDebugInfo(AssemblyDebugInformation assemblyDebugInformation, MethodBase methodBase, MethodDefinitionHandle definitionHandle)
    {
        AssemblyDebugInformation = assemblyDebugInformation;
        MethodBase = methodBase;
        Handle = definitionHandle;
    }

    /// <summary>
    /// Gets the sequence points.
    /// </summary>
    /// <exception cref="InvalidOperationException">Sequence points not loaded.</exception>
    public ImmutableArray<PdbSequencePoint> SequencePoints
    {
        get
        {
            var scopesSuccess = AssemblyDebugInformation.TryLoadSequencePoints(this, out var sequencePoints);

            if (!scopesSuccess)
            {
                throw new InvalidOperationException("Sequence points not loaded.");
            }

            return sequencePoints;
        }
    }

    /// <summary>
    /// Gets the local scopes.
    /// </summary>
    /// <exception cref="InvalidOperationException">Local scopes not loaded.</exception>
    public ImmutableArray<PdbLocalScope> LocalScopes
    {
        get
        {
            var scopesSuccess = AssemblyDebugInformation.TryLoadLocalScopes(this, out var localScopes);

            if (!scopesSuccess)
            {
                throw new InvalidOperationException("Local scopes not loaded.");
            }

            return localScopes;
        }
    }

    /// <summary>
    /// Gets the method base.
    /// </summary>
    public MethodBase MethodBase { get; }

    /// <summary>
    /// Gets the assembly debug information.
    /// </summary>
    public AssemblyDebugInformation AssemblyDebugInformation { get; }

    /// <summary>
    /// Gets the method definition handle.
    /// </summary>
    public MethodDefinitionHandle Handle { get; }

    /// <summary>
    /// Gets the local variables.
    /// </summary>
    public ImmutableArray<PdbLocalVariable> LocalVariables
    {
        get
        {
            var builder = ImmutableArray.CreateBuilder<PdbLocalVariable>();

            foreach (var scope in LocalScopes)
            {
                builder.AddRange(scope.Variables);
            }

            return builder.ToImmutable();
        }
    }
}