// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors.Arithmetic;
using Sci.NET.Mathematics.Tensors.Arithmetic.Implementations;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Common.Implementations;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

namespace Sci.NET.Mathematics.Tensors;

internal class TensorOperationServiceFactory : ITensorOperationServiceFactory
{
    private readonly IMatrixMultiplicationService _matrixMultiplicationService;
    private readonly IDeviceGuardService _deviceGuardService;
    private readonly IPermutationService _permutationService;
    private readonly IReshapeService _reshapeService;
    private readonly IContractionService _contractionService;
    private readonly IArithmeticService _arithmeticService;

    public TensorOperationServiceFactory()
    {
        _reshapeService = new ReshapeService();
        _deviceGuardService = new DeviceGuardService();
        _matrixMultiplicationService = new MatrixMultiplicationService(this);
        _arithmeticService = new ArithmeticService(this);
        _permutationService = new PermutationService(this);
        _contractionService = new ContractionService(this);
    }

    public IMatrixMultiplicationService GetMatrixMultiplicationService()
    {
        return _matrixMultiplicationService;
    }

    public IDeviceGuardService GetDeviceGuardService()
    {
        return _deviceGuardService;
    }

    public IPermutationService GetPermutationService()
    {
        return _permutationService;
    }

    public IReshapeService GetReshapeService()
    {
        return _reshapeService;
    }

    public IContractionService GetContractionService()
    {
        return _contractionService;
    }

    public IArithmeticService GetArithmeticService()
    {
        return _arithmeticService;
    }
}