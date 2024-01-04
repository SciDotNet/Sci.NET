// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.LowLevel;

/// <summary>
/// An exception thrown when an intrinsic type is not implemented.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
public class IntrinsicTypeNotImplementedException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IntrinsicTypeNotImplementedException"/> class.
    /// </summary>
    public IntrinsicTypeNotImplementedException()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IntrinsicTypeNotImplementedException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public IntrinsicTypeNotImplementedException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IntrinsicTypeNotImplementedException"/> class.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public IntrinsicTypeNotImplementedException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}