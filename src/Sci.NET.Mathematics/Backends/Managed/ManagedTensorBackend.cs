// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Backends.Managed;

/// <summary>
/// An implementation of <see cref="ITensorBackend"/> for the managed backend.
/// </summary>
[PublicAPI]
public class ManagedTensorBackend : ITensorBackend
{
    internal const int ParallelizationThreshold = 10000;

    /// <summary>
    /// Initializes a new instance of the <see cref="ManagedTensorBackend"/> class.
    /// </summary>
    public ManagedTensorBackend()
    {
        Storage = new ManagedStorageBackend();
        LinearAlgebra = new ManagedLinearAlgebraBackend();
        Arithmetic = new ManagedArithmeticBackend();
        Device = new CpuComputeDevice();
    }

    /// <summary>
    /// Gets the singleton instance of the <see cref="ManagedTensorBackend"/>.
    /// </summary>
    public static ITensorBackend Instance { get; } = new ManagedTensorBackend();

    /// <inheritdoc />
    public ITensorStorageBackend Storage { get; }

    /// <inheritdoc />
    public ILinearAlgebraBackend LinearAlgebra { get; }

    /// <inheritdoc />
    public IArithmeticBackend Arithmetic { get; }

    /// <inheritdoc />
    public IDevice Device { get; }
}