// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Pointwise;
using Sci.NET.Mathematics.Tensors.Serialization;
using Sci.NET.Mathematics.Tensors.Trigonometry;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface providing methods too build tensor operation services.
/// </summary>
[PublicAPI]
public interface ITensorOperationServiceProvider
{
    /// <summary>
    /// Gets an instance of the <see cref="IMatrixMultiplicationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IMatrixMultiplicationService"/>.</returns>
    public IMatrixMultiplicationService GetMatrixMultiplicationService();

    /// <summary>
    /// Gets an instance of the <see cref="IDeviceGuardService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IDeviceGuardService"/>.</returns>
    public IDeviceGuardService GetDeviceGuardService();

    /// <summary>
    /// Gets an instance of the <see cref="IPermutationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IPermutationService"/>.</returns>
    public IPermutationService GetPermutationService();

    /// <summary>
    /// Gets an instance of the <see cref="IReshapeService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IReshapeService"/>.</returns>
    public IReshapeService GetReshapeService();

    /// <summary>
    /// Gets an instance of the <see cref="IContractionService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IContractionService"/>.</returns>
    public IContractionService GetContractionService();

    /// <summary>
    /// Gets an instance of the <see cref="IArithmeticService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IArithmeticService"/>.</returns>
    public IArithmeticService GetArithmeticService();

    /// <summary>
    /// Gets an instance of the <see cref="IPowerService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IPowerService"/>.</returns>
    public IPowerService GetPowerService();

    /// <summary>
    /// Gets an instance of the <see cref="IReductionService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="IReductionService"/>.</returns>
    public IReductionService GetReductionService();

    /// <summary>
    /// Gets an instance of the <see cref="ILinqService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ILinqService"/>.</returns>
    public ILinqService GetLinqService();

    /// <summary>
    /// Gets an instance of the <see cref="ITrigonometryService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ITrigonometryService"/>.</returns>
    public ITrigonometryService GetTrigonometryService();

    /// <summary>
    /// Gets an instance the <see cref="ISerializationService"/>.
    /// </summary>
    /// <returns>An instance of the <see cref="ISerializationService"/>.</returns>
    public ISerializationService GetSerializationService();
}