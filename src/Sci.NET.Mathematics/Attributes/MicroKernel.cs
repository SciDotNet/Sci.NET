// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Mathematics.Attributes;

/// <summary>
/// An attribute to mark a class as a micro kernel.
/// </summary>
[ExcludeFromCodeCoverage]
[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
public sealed class MicroKernelAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MicroKernelAttribute"/> class.
    /// </summary>
    public MicroKernelAttribute()
    {
    }
}