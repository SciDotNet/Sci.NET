// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

/// <summary>
/// A provider for debug symbols.
/// </summary>
[PublicAPI]
public interface IDebugSymbolProvider
{
    /// <summary>
    /// Gets the method variables.
    /// </summary>
    /// <param name="methodBase">The method base.</param>
    /// <returns>The method variables.</returns>
    public LocalVariable[] GetMethodVariables(MethodBase methodBase);
}