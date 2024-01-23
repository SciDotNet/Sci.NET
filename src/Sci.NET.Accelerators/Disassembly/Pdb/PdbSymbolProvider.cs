// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Reflection;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// A PDB provider.
/// </summary>
[PublicAPI]
public sealed class PdbSymbolProvider : IDebugSymbolProvider
{
    private static readonly ConcurrentDictionary<string, IDebugSymbolProvider> MetadataReaders = new ();

    /// <summary>
    /// Reads a PDB file for the given assembly.
    /// </summary>
    /// <param name="assembly">The assembly.</param>
    /// <returns>The metadata reader.</returns>
    public static IDebugSymbolProvider ReadPdbFile(Assembly assembly)
    {
        if (MetadataReaders.TryGetValue(assembly.Location, out var provider))
        {
            return provider;
        }

        var info = new FakeSymbolProvider();

        _ = MetadataReaders.TryAdd(assembly.Location, info);

        return info;
    }

    /// <inheritdoc />
    public LocalVariable[] GetMethodVariables(MethodBase methodBase)
    {
        var variables = new Dictionary<int, string>();

        return methodBase
                   .GetMethodBody()
                   ?.LocalVariables.Select(x => new LocalVariable { Index = x.LocalIndex, Name = variables[x.LocalIndex], Type = x.LocalType })
                   .ToArray() ??
               throw new InvalidOperationException("No local variables found.");
    }
}