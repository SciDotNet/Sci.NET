// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;

namespace Sci.NET.CUDA.CuBLAS.Types;

/// <summary>
/// Represents an opaque handle to a CUBLAS library context.
/// </summary>
[PublicAPI]
[StructLayout(LayoutKind.Explicit)]
[DebuggerDisplay("{_handle}")]
public readonly struct OpaqueCublasHandle : IValueEquatable<OpaqueCublasHandle>
{
    [FieldOffset(0)] private readonly nuint _handle;

    /// <summary>
    /// Initializes a new instance of the <see cref="OpaqueCublasHandle"/> struct.
    /// </summary>
    /// <param name="handle">The handle to the CuBLAS library context.</param>
    public unsafe OpaqueCublasHandle(void* handle)
    {
        _handle = (nuint)handle;
    }

    /// <inheritdoc />
    public static bool operator ==(OpaqueCublasHandle left, OpaqueCublasHandle right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(OpaqueCublasHandle left, OpaqueCublasHandle right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(OpaqueCublasHandle other)
    {
        return _handle == other._handle;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is OpaqueCublasHandle other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return _handle.GetHashCode();
    }
}