// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Reflection.Metadata;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

internal class FakeAssemblyDebugInformation : IAssemblyDebugInformation
{
    private readonly Dictionary<MethodBase, MethodDebugInfo> _methodDebugInformation;

    /// <summary>
    /// Initializes a new instance of the <see cref="FakeAssemblyDebugInformation"/> class.
    /// </summary>
    /// <param name="assembly">The assembly.</param>
    public FakeAssemblyDebugInformation(Assembly assembly)
    {
        Assembly = assembly;
        Modules = ImmutableArray.Create(assembly.GetModules());
        _methodDebugInformation = new Dictionary<MethodBase, MethodDebugInfo>();

        foreach (var methodHandle in assembly.GetModules().SelectMany(x => x.GetMethods()))
        {
            _methodDebugInformation.Add(methodHandle, new MethodDebugInfo(this, methodHandle, default));
        }
    }

    public Assembly Assembly { get; }

    public MetadataReader MetadataReader => throw new InvalidOperationException();

    public ImmutableArray<Module> Modules { get; }

    public bool TryGetMethodDebugInfo(MethodBase methodBase, [NotNullWhen(true)] out MethodDebugInfo? methodDebugInfo)
    {
        if (methodBase.GetGenericArguments().Length > 0 && methodBase is MethodInfo methodInfo)
        {
            methodBase = methodInfo.GetGenericMethodDefinition();
        }

        return _methodDebugInformation.TryGetValue(methodBase, out methodDebugInfo);
    }

    public bool TryLoadSequencePoints(MethodDebugInfo methodDebugInfo, out ImmutableArray<PdbSequencePoint> sequencePoints)
    {
        sequencePoints = ImmutableArray<PdbSequencePoint>.Empty;

        return true;
    }

    public bool TryLoadLocalScopes(MethodDebugInfo methodDebugInfo, out ImmutableArray<PdbLocalScope> localScopes)
    {
        var methodBody = methodDebugInfo.MethodBase.GetMethodBody();
        var variables = methodBody
            ?.LocalVariables.Select(
                x => new PdbLocalVariable
                {
                    Index = x.LocalIndex,
                    Name = $"fake_local_{x.LocalIndex}"
                })
            .ToImmutableArray() ?? ImmutableArray<PdbLocalVariable>.Empty;

        localScopes = new ImmutableArray<PdbLocalScope>
        {
            new()
            {
                StartOffset = 0,
                EndOffset = methodBody?.GetILAsByteArray()?.Length ?? 0,
                Variables = variables
            }
        };
        return true;
    }

    public void Dispose()
    {
        // Nothing to dispose
    }
}