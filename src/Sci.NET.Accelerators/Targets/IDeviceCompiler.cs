// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.IR.Rewriter;

namespace Sci.NET.Accelerators.Targets;

/// <summary>
/// Represents a device compiler.
/// </summary>
[PublicAPI]
public interface IDeviceCompiler
{
    /// <summary>
    /// Compiles the given method.
    /// </summary>
    /// <param name="method">The method to compile.</param>
    /// <returns>The compiled kernel.</returns>
    public ICompiledKernel Compile(SsaMethod method);
}