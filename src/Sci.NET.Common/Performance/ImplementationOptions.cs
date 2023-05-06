// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;

namespace Sci.NET.Common.Performance;

/// <summary>
/// Provides constants for <see cref="MethodImplOptions"/> defaults.
/// </summary>
[PublicAPI]
public static class ImplementationOptions
{
    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to hint for inline, optimized
    /// and unmanaged code to be generated.
    /// </summary>
    public const MethodImplOptions UnmanagedHotPath = MethodImplOptions.AggressiveInlining |
                                                      MethodImplOptions.AggressiveOptimization |
                                                      MethodImplOptions.Unmanaged;

    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to hint for inline, optimized code
    /// to be generated.
    /// </summary>
    public const MethodImplOptions HotPath = MethodImplOptions.AggressiveInlining |
                                             MethodImplOptions.AggressiveOptimization;

    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to hint for inline code to be generated.
    /// </summary>
    public const MethodImplOptions FastPath = MethodImplOptions.AggressiveInlining;

    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to hint for indicating the method body
    /// is written in unmanaged code.
    /// </summary>
    public const MethodImplOptions Unmanaged = MethodImplOptions.Unmanaged;

    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to disable inlining.
    /// </summary>
    public const MethodImplOptions NoInlining = MethodImplOptions.NoInlining;

    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to disable optimization.
    /// </summary>
    public const MethodImplOptions NoOptimization = MethodImplOptions.NoOptimization;

    /// <summary>
    /// Gets <see cref="MethodImplOptions"/> to hint for optimized code to be generated.
    /// </summary>
    public const MethodImplOptions AggressiveOptimization = MethodImplOptions.AggressiveOptimization;
}