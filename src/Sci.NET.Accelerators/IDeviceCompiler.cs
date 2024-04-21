// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Rewriter;

namespace Sci.NET.Accelerators;

/// <summary>
/// Represents a device compiler.
/// </summary>
/// <typeparam name="TKernel">The type of the compiled kernel.</typeparam>
[PublicAPI]
public interface IDeviceCompiler<out TKernel>
    where TKernel : ICompiledKernel
{
    /// <summary>
    /// Gets the identifier of the compiler.
    /// </summary>
#pragma warning disable CA1000
    public static abstract Guid Identifier { get; }
#pragma warning restore CA1000

    /// <summary>
    /// Compiles the intermediate representation into a kernel.
    /// </summary>
    /// <param name="intermediateRepresentation">The intermediate representation to compile.</param>
    /// <returns>The compiled kernel.</returns>
    public TKernel Compile(MsilSsaMethod intermediateRepresentation);
}