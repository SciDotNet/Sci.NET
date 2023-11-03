// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.NeuralNetworks;
using Sci.NET.Mathematics.Tensors.Pointwise;
using Sci.NET.Mathematics.Tensors.Random;
using Sci.NET.Mathematics.Tensors.Serialization;
using Sci.NET.Mathematics.Tensors.Trigonometry;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface providing methods too build tensor operation services.
/// </summary>
[PublicAPI]
public interface ITensorOperationServiceProvider
{
#pragma warning disable CA1024
    /// <summary>
    /// Gets an instance of the <see cref="IMatrixMultiplicationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IMatrixMultiplicationService"/>.</returns>
    [DebuggerStepThrough]
    public IMatrixMultiplicationService GetMatrixMultiplicationService();

    /// <summary>
    /// Gets an instance of the <see cref="IDeviceGuardService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IDeviceGuardService"/>.</returns>
    [DebuggerStepThrough]
    public IDeviceGuardService GetDeviceGuardService();

    /// <summary>
    /// Gets an instance of the <see cref="IPermutationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IPermutationService"/>.</returns>
    [DebuggerStepThrough]
    public IPermutationService GetPermutationService();

    /// <summary>
    /// Gets an instance of the <see cref="IReshapeService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IReshapeService"/>.</returns>
    [DebuggerStepThrough]
    public IReshapeService GetReshapeService();

    /// <summary>
    /// Gets an instance of the <see cref="IContractionService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IContractionService"/>.</returns>
    [DebuggerStepThrough]
    public IContractionService GetContractionService();

    /// <summary>
    /// Gets an instance of the <see cref="IArithmeticService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IArithmeticService"/>.</returns>
    [DebuggerStepThrough]
    public IArithmeticService GetArithmeticService();

    /// <summary>
    /// Gets an instance of the <see cref="IPowerService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IPowerService"/>.</returns>
    [DebuggerStepThrough]
    public IPowerService GetPowerService();

    /// <summary>
    /// Gets an instance of the <see cref="IReductionService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IReductionService"/>.</returns>
    [DebuggerStepThrough]
    public IReductionService GetReductionService();

    /// <summary>
    /// Gets an instance of the <see cref="ILinqService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ILinqService"/>.</returns>
    [DebuggerStepThrough]
    public ILinqService GetLinqService();

    /// <summary>
    /// Gets an instance of the <see cref="ITrigonometryService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ITrigonometryService"/>.</returns>
    [DebuggerStepThrough]
    public ITrigonometryService GetTrigonometryService();

    /// <summary>
    /// Gets an instance the <see cref="ISerializationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ISerializationService"/>.</returns>
    [DebuggerStepThrough]
    public ISerializationService GetSerializationService();

    /// <summary>
    /// Gets an instance of the <see cref="ICastingService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ICastingService"/>.</returns>
    [DebuggerStepThrough]
    public ICastingService GetCastingService();

    /// <summary>
    /// Gets an instance of the <see cref="IConvolutionService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IConvolutionService"/>.</returns>
    [DebuggerStepThrough]
    public IConvolutionService GetConvolutionService();

    /// <summary>
    /// Gets an instance of the <see cref="IConcatenationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IConvolutionService"/>.</returns>
    [DebuggerStepThrough]
    public IConcatenationService GetConcatenationService();

    /// <summary>
    /// Gets an instance of the <see cref="IActivationFunctionService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IActivationFunctionService"/>.</returns>
    [DebuggerStepThrough]
    public IActivationFunctionService GetActivationFunctionService();

    /// <summary>
    /// Gets an instance of the <see cref="IBroadcastService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IBroadcastService"/>.</returns>
    [DebuggerStepThrough]
    public IBroadcastService GetBroadcastingService();

    /// <summary>
    /// Gets an instance of the <see cref="IVectorOperationsService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IVectorOperationsService"/>.</returns>
    [DebuggerStepThrough]
    public IVectorOperationsService GetVectorOperationsService();

    /// <summary>
    /// Gets an instance of the <see cref="IRandomService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IRandomService"/>.</returns>
    [DebuggerStepThrough]
    public IRandomService GetRandomService();

#pragma warning restore CA1024
}