// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// An exception thrown when an operation is not supported by the automatic differentiation system.
/// </summary>
public class AutoDiffNotSupportedException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AutoDiffNotSupportedException"/> class.
    /// </summary>
    /// <param name="operationName">The name of the operation that is not supported.</param>
    public AutoDiffNotSupportedException(string operationName)
        : base($"The operation '{operationName}' does not support automatic differentiation.")
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="AutoDiffNotSupportedException"/> class.
    /// </summary>
    /// <param name="operationName">The name of the operation that is not supported.</param>
    /// <param name="inner">The inner exception.</param>
    public AutoDiffNotSupportedException(string operationName, Exception inner)
        : base($"The operation '{operationName}' does not support automatic differentiation.", inner)
    {
    }
}