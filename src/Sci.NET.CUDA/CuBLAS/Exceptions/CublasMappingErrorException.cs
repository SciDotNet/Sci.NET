// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception thrown when accessing GPU memory fails.
/// </summary>
[PublicAPI]
public class CublasMappingErrorException : Exception
{
    private const string DefaultMessage =
        "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasMappingErrorException"/> class.
    /// </summary>
    public CublasMappingErrorException()
        : base(DefaultMessage)
    {
    }
}