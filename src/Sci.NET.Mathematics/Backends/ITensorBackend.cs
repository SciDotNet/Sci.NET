// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for tensor backends.
/// </summary>
[PublicAPI]
public interface ITensorBackend
{
    /// <summary>
    /// Gets the <see cref="ITensorStorageBackend"/> implementation for the <see cref="ITensorBackend"/>.
    /// </summary>
    public ITensorStorageBackend Storage { get; }

    /// <summary>
    /// Gets the <see cref="ILinearAlgebraBackend"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public ILinearAlgebraBackend LinearAlgebra { get; }

    /// <summary>
    /// Gets the <see cref="IArithmeticBackend"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IArithmeticBackend Arithmetic { get; }

    /// <summary>
    /// Gets the <see cref="IDevice"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IDevice Device { get; }
}