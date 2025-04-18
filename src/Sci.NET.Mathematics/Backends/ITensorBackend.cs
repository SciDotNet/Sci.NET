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
    /// Gets the <see cref="ITensorStorageKernels"/> implementation for the <see cref="ITensorBackend"/>.
    /// </summary>
    public ITensorStorageKernels Storage { get; }

    /// <summary>
    /// Gets the <see cref="ILinearAlgebraKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public ILinearAlgebraKernels LinearAlgebra { get; }

    /// <summary>
    /// Gets the <see cref="IArithmeticKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IArithmeticKernels Arithmetic { get; }

    /// <summary>
    /// Gets the <see cref="IPowerKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IPowerKernels Power { get; }

    /// <summary>
    /// Gets the <see cref="IDevice"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IDevice Device { get; }

    /// <summary>
    /// Gets the <see cref="IReductionKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IReductionKernels Reduction { get; }

    /// <summary>
    /// Gets the <see cref="ITrigonometryKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public ITrigonometryKernels Trigonometry { get; }

    /// <summary>
    /// Gets the <see cref="IRandomKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IRandomKernels Random { get; }

    /// <summary>
    /// Gets the <see cref="ICastingKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public ICastingKernels Casting { get; }

    /// <summary>
    /// Gets the <see cref="IActivationFunctionKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IActivationFunctionKernels ActivationFunctions { get; }

    /// <summary>
    /// Gets the <see cref="IBroadcastingKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IBroadcastingKernels Broadcasting { get; }

    /// <summary>
    /// Gets the <see cref="IPermutationKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IPermutationKernels Permutation { get; }

    /// <summary>
    /// Gets the <see cref="INormalisationKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public INormalisationKernels Normalisation { get; }

    /// <summary>
    /// Gets the <see cref="IEqualityOperationKernels"/> instance for the <see cref="ITensorBackend"/>.
    /// </summary>
    public IEqualityOperationKernels EqualityOperations { get; }
}