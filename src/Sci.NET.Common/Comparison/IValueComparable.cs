// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Comparison;

/// <summary>
/// Provides an interface for value comparison of two objects.
/// </summary>
/// <typeparam name="TLeft">The left type to compare.</typeparam>
/// <typeparam name="TRight">The right type to compare.</typeparam>
[PublicAPI]
[SuppressMessage("Usage", "CA2225:Operator overloads have named alternates", Justification = "Architectural decision.")]
public interface IValueComparable<in TLeft, in TRight>
    where TLeft : IValueComparable<TLeft, TRight>
    where TRight : IValueComparable<TLeft, TRight>
{
    /// <summary>
    /// Determines if <paramref name="left"/> is less than <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>
    /// <c>true</c> if <paramref name="left"/> is less than
    /// <paramref name="right"/>, else <c>false</c>.
    /// </returns>
    public static abstract bool operator <(TLeft left, TRight right);

    /// <summary>
    /// Determines if <paramref name="left"/> is less than or equal to
    /// <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>
    /// <c>true</c> if <paramref name="left"/> is less than or equal to
    /// <paramref name="right"/>, else <c>false</c>.
    /// </returns>
    public static abstract bool operator <=(TLeft left, TRight right);

    /// <summary>
    /// Determines if <paramref name="left"/> is greater than <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>
    /// <c>true</c> if <paramref name="left"/> is greater than
    /// <paramref name="right"/>, else <c>false</c>.
    /// </returns>
    public static abstract bool operator >(TLeft left, TRight right);

    /// <summary>
    /// Determines if <paramref name="left"/> is greater than
    /// or equal to <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>
    /// <c>true</c> if <paramref name="left"/> is greater than or equal to
    /// <paramref name="right"/>, else <c>false</c>.
    /// </returns>
    public static abstract bool operator >=(TLeft left, TRight right);
}