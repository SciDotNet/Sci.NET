// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators;

[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
internal readonly struct KernelCacheEntry : IValueEquatable<KernelCacheEntry>
{
    public required MethodInfo Method { get; init; }

    public required Guid CompilerIdentifier { get; init; }

    public static bool operator ==(KernelCacheEntry left, KernelCacheEntry right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(KernelCacheEntry left, KernelCacheEntry right)
    {
        return !(left == right);
    }

    public bool Equals(KernelCacheEntry other)
    {
        return Method.Equals(other.Method) && CompilerIdentifier.Equals(other.CompilerIdentifier);
    }

    public override bool Equals(object? obj)
    {
        return obj is KernelCacheEntry other && Equals(other);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Method, CompilerIdentifier);
    }
}