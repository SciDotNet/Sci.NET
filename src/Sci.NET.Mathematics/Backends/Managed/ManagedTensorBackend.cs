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
        Storage = new ManagedStorageKernels();
        LinearAlgebra = new ManagedLinearAlgebraKernels();
        Arithmetic = new ManagedArithmeticKernels();
        Power = new ManagedPowerKernels();
        Device = new CpuComputeDevice();
        Reduction = new ManagedReductionKernels();
        Linq = new ManagedLinqKernels();
        Trigonometry = new ManagedTrigonometryKernels();
        Random = new ManagedRandomKernels();
        Casting = new ManagedCastingKernels();
        NeuralNetworks = new ManagedNeuralNetworkKernels();
        ActivationFunctions = new ManagedActivationFunctionKernels();
    }

    /// <summary>
    /// Gets the singleton instance of the <see cref="ManagedTensorBackend"/>.
    /// </summary>
    public static ITensorBackend Instance { get; } = new ManagedTensorBackend();

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
}