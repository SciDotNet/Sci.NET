// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators;

/// <summary>
/// Represents a compiled kernel.
/// </summary>
[PublicAPI]
public interface ICompiledKernel
{
    /// <summary>
    /// Gets the name of the kernel.
    /// </summary>
    public string Name { get; }
}