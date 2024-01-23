// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// A provider for debug symbols when no PDB is available.
/// </summary>
public class FakeSymbolProvider : IDebugSymbolProvider
{
    /// <inheritdoc />
    public LocalVariable[] GetMethodVariables(MethodBase methodBase)
    {
        return (methodBase.GetMethodBody()?.LocalVariables ?? throw new InvalidOperationException("No local variables found.")).Select(x => new LocalVariable
        {
            Index = x.LocalIndex,
            Name = $"local_{x.LocalIndex}",
            Type = x.LocalType
        }).ToArray();
    }
}