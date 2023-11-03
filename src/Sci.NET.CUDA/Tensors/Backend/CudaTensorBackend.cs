// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.CUDA.Tensors.Backend;

/// <summary>
/// Represents a CUDA tensor backend.
/// </summary>
[PublicAPI]
public class CudaTensorBackend : ITensorBackend
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CudaTensorBackend"/> class.
    /// </summary>
    public CudaTensorBackend()
    {
        Device = new CudaComputeDevice();
        Storage = new CudaStorageKernels();
        LinearAlgebra = new CudaLinearAlgebraKernels();
        Trigonometry = new CudaTrigonometryKernels();
        Arithmetic = new CudaArithmeticKernels();
        Random = new CudaRandomKernels();
    }

    /// <inheritdoc />
    public ITensorStorageKernels Storage { get; }

    /// <inheritdoc />
    public ILinearAlgebraKernels LinearAlgebra { get; }

    /// <inheritdoc />
    public IArithmeticKernels Arithmetic { get; }

    /// <inheritdoc />
    public IPowerKernels Power { get; } = null!;

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public IReductionKernels Reduction { get; } = null!;

    /// <inheritdoc />
    public ILinqKernels Linq { get; } = null!;

    /// <inheritdoc />
    public ITrigonometryKernels Trigonometry { get; }

    /// <inheritdoc />
    public IRandomKernels Random { get; }

    /// <inheritdoc />
    public ICastingKernels Casting { get; } = null!;

    /// <inheritdoc />
    public INeuralNetworkKernels NeuralNetworks { get; } = null!;

    /// <inheritdoc />
    public IActivationFunctionKernels ActivationFunctions { get; } = null!;

    /// <inheritdoc />
    public IBroadcastingKernels Broadcasting { get; } = null!;
}