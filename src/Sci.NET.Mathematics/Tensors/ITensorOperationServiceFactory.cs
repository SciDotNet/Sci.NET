// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors.Arithmetic;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.Manipulation;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface providing methods too build tensor operation services.
/// </summary>
[PublicAPI]
public interface ITensorOperationServiceFactory
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
}