// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Reflection.Metadata;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// Represents the debug information for an assembly.
/// </summary>
[PublicAPI]
public interface IAssemblyDebugInformation : IDisposable
{
    /// <summary>
    /// Gets the assembly.
    /// </summary>
    public Assembly Assembly { get; }

    /// <summary>
    /// Gets the metadata reader.
    /// </summary>
    public MetadataReader MetadataReader { get; }

    /// <summary>
    /// Gets the modules.
    /// </summary>
    public ImmutableArray<Module> Modules { get; }

    /// <summary>
    /// Gets a value indicating whether the debug information is valid.
    /// </summary>
    /// <param name="methodBase">The method base.</param>
    /// <param name="methodDebugInfo">The method debug information.</param>
    /// <returns>A value indicating whether the debug information is valid.</returns>
    public bool TryGetMethodDebugInfo(MethodBase methodBase, [NotNullWhen(true)] out MethodDebugInfo? methodDebugInfo);

    /// <summary>
    /// Tries to load the sequence points.
    /// </summary>
    /// <param name="methodDebugInfo">The method debug information.</param>
    /// <param name="sequencePoints">The sequence points.</param>
    /// <returns>A value indicating whether the sequence points were loaded.</returns>
    public bool TryLoadSequencePoints(MethodDebugInfo methodDebugInfo, out ImmutableArray<PdbSequencePoint> sequencePoints);

    /// <summary>
    /// Tries to load the local scopes.
    /// </summary>
    /// <param name="methodDebugInfo">The method debug information.</param>
    /// <param name="localScopes">The local scopes.</param>
    /// <returns>A value indicating whether the local scopes were loaded.</returns>
    public bool TryLoadLocalScopes(MethodDebugInfo methodDebugInfo, out ImmutableArray<PdbLocalScope> localScopes);
}