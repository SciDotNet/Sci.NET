// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Attributes;

/// <summary>
/// Annotates a method as a kernel method.
/// </summary>
[PublicAPI]
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
public sealed class KernelAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="KernelAttribute"/> class.
    /// </summary>
    /// <param name="outputName">The name of the kernel.</param>
    public KernelAttribute(string? outputName = null)
    {
        OutputName = outputName ?? string.Empty;
    }

    /// <summary>
    /// Gets the name of the kernel.
    /// </summary>
    public string OutputName { get; }
}