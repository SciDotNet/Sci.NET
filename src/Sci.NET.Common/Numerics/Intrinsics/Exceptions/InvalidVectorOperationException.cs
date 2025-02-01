// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

namespace Sci.NET.Common.Numerics.Intrinsics.Exceptions;

/// <summary>
/// An exception thrown when an invalid operation is performed on a vector.
/// </summary>
public class InvalidVectorOperationException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidVectorOperationException"/> class.
    /// </summary>
    public InvalidVectorOperationException()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidVectorOperationException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public InvalidVectorOperationException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidVectorOperationException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="inner">The exception that is the cause of the current exception.</param>
    public InvalidVectorOperationException(string message, Exception inner)
        : base(message, inner)
    {
    }

    /// <summary>
    /// Throws an exception if the left and right vectors are not of the given type.
    /// </summary>
    /// <param name="left">The left vector.</param>
    /// <param name="right">The right vector.</param>
    /// <typeparam name="T">The type of the vector.</typeparam>
    /// <typeparam name="TNumber">The number type of the vector.</typeparam>
    /// <exception cref="InvalidVectorOperationException">Thrown when the left or right vector is not of the given type.</exception>
    [StackTraceHidden]
    public static void ThrowIfNotOfType<T, TNumber>(ISimdVector<TNumber> left, ISimdVector<TNumber> right)
        where T : ISimdVector<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left is not T)
        {
            throw new InvalidVectorOperationException($"The left vector is not of type {typeof(T).Name}.");
        }

        if (right is not T)
        {
            throw new InvalidVectorOperationException($"The right vector is not of type {typeof(T).Name}.");
        }
    }

    /// <summary>
    /// Throws an exception if the input vector is not of the given type.
    /// </summary>
    /// <param name="input">The left vector.</param>
    /// <typeparam name="T">The type of the vector.</typeparam>
    /// <typeparam name="TNumber">The number type of the vector.</typeparam>
    /// <exception cref="InvalidVectorOperationException">Thrown when the left or right vector is not of the given type.</exception>
    [StackTraceHidden]
    public static void ThrowIfNotOfType<T, TNumber>(ISimdVector<TNumber> input)
        where T : ISimdVector<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (input is not T)
        {
            throw new InvalidVectorOperationException($"The left vector is not of type {typeof(T).Name}.");
        }
    }
}