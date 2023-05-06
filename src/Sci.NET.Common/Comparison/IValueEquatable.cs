// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Comparison;

/// <inheritdoc cref="System.IEquatable{T}"/>
[PublicAPI]
public interface IValueEquatable<T> : IEquatable<T>
    where T : struct, IValueEquatable<T>
{
    /// <summary>
    /// Determines whether the specified object is equal to the current object.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns><c>true</c> if <paramref name="left"/> is equal to
    /// <paramref name="right"/>, else <c>false</c>.</returns>
    public static abstract bool operator ==(T left, T right);

    /// <summary>
    /// Determines whether the specified object is equal to the current object.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns><c>true</c> if <paramref name="left"/> is not
    /// equal to <paramref name="right"/>, else <c>false</c>.</returns>
    public static abstract bool operator !=(T left, T right);

    /// <summary>
    /// Determines whether the specified object is equal to the current object.
    /// </summary>
    /// <param name="other">The object to compare to.</param>
    /// <returns><c>true</c> if the <paramref name="other"/>
    /// is equal to <c>this</c> instance, else <c>false</c>.</returns>
    public new bool Equals(T other);

    /// <inheritdoc cref="object.Equals(object?)"/>
    public bool Equals(object? obj);

    /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
    [ExcludeFromCodeCoverage]
    bool IEquatable<T>.Equals(T other)
    {
        return Equals(other);
    }

    /// <inheritdoc cref="object.GetHashCode"/>
    public int GetHashCode();
}