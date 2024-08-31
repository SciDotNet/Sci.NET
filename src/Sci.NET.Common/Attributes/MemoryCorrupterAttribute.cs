// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Attributes;

/// <summary>
/// An attribute which decorates a method which performs memory manipulations without any checks. It doesn't do anything
/// other than remind you that calling this method (without caution) will cause you to have a bad day.
/// </summary>
[PublicAPI]
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
[ExcludeFromCodeCoverage]
public sealed class MemoryCorrupterAttribute : Attribute
{
}