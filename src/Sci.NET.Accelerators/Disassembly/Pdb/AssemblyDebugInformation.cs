// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

internal sealed class AssemblyDebugInformation : IAssemblyDebugInformation, IDisposable
{
    private readonly Dictionary<MethodBase, MethodDebugInfo> _methodDebugInformation;
    private readonly MetadataReaderProvider _readerProvider;

    public AssemblyDebugInformation(Assembly assembly, Stream pdbStream)
    {
        Assembly = assembly;
        Modules = ImmutableArray.Create(assembly.GetModules());
        _methodDebugInformation = new Dictionary<MethodBase, MethodDebugInfo>();
        _readerProvider = MetadataReaderProvider.FromPortablePdbStream(pdbStream, MetadataStreamOptions.Default);
        MetadataReader = _readerProvider.GetMetadataReader();

        foreach (var methodHandle in MetadataReader.MethodDebugInformation)
        {
            var definitionHandle = methodHandle.ToDefinitionHandle();
            var metadataToken = MetadataTokens.GetToken(definitionHandle);

            if (TryResolveMethod(metadataToken, out MethodBase? method))
            {
                _methodDebugInformation.Add(
                    method,
                    new MethodDebugInfo(
                        this,
                        method,
                        definitionHandle));
            }
        }
    }

    public Assembly Assembly { get; }

    public MetadataReader MetadataReader { get; }

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
        var sequencePointsBuilder = ImmutableArray.CreateBuilder<PdbSequencePoint>();

        foreach (var sequencePoint in MetadataReader.GetMethodDebugInformation(methodDebugInfo.Handle).GetSequencePoints())
        {
            var document = MetadataReader.GetDocument(sequencePoint.Document);
            var documentName = MetadataReader.GetString(document.Name);

            sequencePointsBuilder.Add(
                new PdbSequencePoint
                {
                    Offset = sequencePoint.Offset,
                    DocumentName = documentName,
                    StartLine = sequencePoint.StartLine,
                    StartColumn = sequencePoint.StartColumn,
                    EndLine = sequencePoint.EndLine,
                    EndColumn = sequencePoint.EndColumn
                });
        }

        sequencePoints = sequencePointsBuilder.ToImmutableArray();
        return true;
    }

    public bool TryLoadLocalScopes(MethodDebugInfo methodDebugInfo, out ImmutableArray<PdbLocalScope> localScopes)
    {
        var localScopeBuilder = ImmutableArray.CreateBuilder<PdbLocalScope>();
        var scopeIndex = 0;

        foreach (var localScopeHandle in MetadataReader.GetLocalScopes(methodDebugInfo.Handle))
        {
            var scope = MetadataReader.GetLocalScope(localScopeHandle);
            var localVariableBuilder = ImmutableArray.CreateBuilder<PdbLocalVariable>();
            scopeIndex++;

            foreach (var localVariableHandle in scope.GetLocalVariables())
            {
                var localVariable = MetadataReader.GetLocalVariable(localVariableHandle);
                var variableName = localVariable.Name.IsNil ? string.Empty : MetadataReader.GetString(localVariable.Name);

                localVariableBuilder.Add(new PdbLocalVariable { Index = localVariable.Index, Name = $"{variableName}_s{scopeIndex}" });
            }

            localScopeBuilder.Add(new PdbLocalScope { StartOffset = scope.StartOffset, EndOffset = scope.EndOffset, Variables = localVariableBuilder.ToImmutable() });
        }

        localScopes = localScopeBuilder.ToImmutableArray();
        return true;
    }

    public void Dispose()
    {
        _readerProvider.Dispose();
    }

    private bool TryResolveMethod(int metadataToken, [NotNullWhen(true)] out MethodBase? methodBase)
    {
        foreach (var module in Modules)
        {
            methodBase = module.ResolveMethod(metadataToken);

            if (methodBase is not null)
            {
                return true;
            }
        }

        methodBase = null;
        return false;
    }
}