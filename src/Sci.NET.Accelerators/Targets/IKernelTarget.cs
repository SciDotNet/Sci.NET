// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Targets;

/// <summary>
/// Represents a kernel target.
/// </summary>
[PublicAPI]
public interface IKernelTarget
{
    /// <summary>
    /// Executes the kernel.
    /// </summary>
    public void Execute();
}