// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Reflection;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Rewriter;

namespace Sci.NET.Accelerators;

/// <summary>
/// Represents a compiler for the Sci.NET Accelerators.
/// </summary>
[PublicAPI]
public static class DeviceCompiler
{
    private static readonly ConcurrentDictionary<KernelCacheEntry, ICompiledKernel> KernelCache = new();

    /// <summary>
    /// Compiles a kernel method.
    /// </summary>
    /// <typeparam name="TCompiler">The type of the compiler to use.</typeparam>
    /// <typeparam name="TKernel">The type of the kernel to compile.</typeparam>
    /// <param name="kernelMethod">The kernel method to compile.</param>
    /// <returns>The compiled kernel.</returns>
    public static TKernel Compile<TCompiler, TKernel>(Action kernelMethod)
        where TCompiler : IDeviceCompiler<TKernel>, new()
        where TKernel : ICompiledKernel
    {
        var method = kernelMethod.Method;

        return Compile<TCompiler, TKernel>(method);
    }

    /// <summary>
    /// Compiles a kernel method.
    /// </summary>
    /// <param name="methodInfo">The method to compile.</param>
    /// <typeparam name="TCompiler">The type of the compiler to use.</typeparam>
    /// <typeparam name="TKernel">The type of the kernel to compile.</typeparam>
    /// <returns>The compiled kernel.</returns>
    public static TKernel Compile<TCompiler, TKernel>(MethodInfo methodInfo)
        where TCompiler : IDeviceCompiler<TKernel>, new()
        where TKernel : ICompiledKernel
    {
        var kernelCacheEntry = new KernelCacheEntry
        {
            Method = methodInfo,
            CompilerIdentifier = TCompiler.Identifier
        };

        if (KernelCache.TryGetValue(kernelCacheEntry, out var kernel) && kernel is TKernel cachedKernel)
        {
            return cachedKernel;
        }

        var disassembledMethod = new MsilDisassembler(new MsilMethodMetadata(methodInfo))
            .Disassemble();

        var intermediateRepresentation = new MsilToIrTranslator(disassembledMethod)
            .Transform();

        var deviceCompiler = new TCompiler();

        var compiledKernel = deviceCompiler.Compile(intermediateRepresentation);

        _ = KernelCache.TryAdd(kernelCacheEntry, compiledKernel);

        return compiledKernel;
    }
}