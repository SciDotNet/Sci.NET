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
        Storage = new CudaStorageKernels();
        LinearAlgebra = new CudaLinearAlgebraKernels();
        Device = new CudaComputeDevice();
        Trigonometry = new CudaTrigonometryKernels();
        Arithmetic = new CudaArithmeticKernels();
    }

    /// <inheritdoc />
    public ITensorStorageKernels Storage { get; }

    /// <inheritdoc />
    public ILinearAlgebraKernels LinearAlgebra { get; }

    /// <inheritdoc />
    public IArithmeticKernels Arithmetic { get; }

    /// <inheritdoc />
    public IPowerKernels Power { get; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public IReductionKernels Reduction { get; }

    /// <inheritdoc />
    public ILinqKernels Linq { get; }

    /// <inheritdoc />
    public ITrigonometryKernels Trigonometry { get; }

    /// <inheritdoc />
    public IRandomKernels Random { get; }

    /// <inheritdoc />
    public ICastingKernels Casting { get; }

    /// <inheritdoc />
    public INeuralNetworkKernels NeuralNetworks { get; }

    /// <inheritdoc />
    public IActivationFunctionKernels ActivationFunctions { get; }

    /// <inheritdoc />
    public IBroadcastingKernels Broadcasting { get; }
}