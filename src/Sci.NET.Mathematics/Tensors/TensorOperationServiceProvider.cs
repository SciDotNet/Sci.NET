// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Common.Implementations;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Manipulation.Implementations;
using Sci.NET.Mathematics.Tensors.Pointwise;
using Sci.NET.Mathematics.Tensors.Pointwise.Implementations;
using Sci.NET.Mathematics.Tensors.Serialization;
using Sci.NET.Mathematics.Tensors.Serialization.Implementations;
using Sci.NET.Mathematics.Tensors.Trigonometry;
using Sci.NET.Mathematics.Tensors.Trigonometry.Implementations;

namespace Sci.NET.Mathematics.Tensors;

internal class TensorOperationServiceProvider : ITensorOperationServiceProvider
{
    private readonly IMatrixMultiplicationService _matrixMultiplicationService;
    private readonly IDeviceGuardService _deviceGuardService;
    private readonly IPermutationService _permutationService;
    private readonly IReshapeService _reshapeService;
    private readonly IContractionService _contractionService;
    private readonly IArithmeticService _arithmeticService;
    private readonly IPowerService _powerService;
    private readonly IReductionService _reductionService;
    private readonly ILinqService _linqService;
    private readonly ITrigonometryService _trigonometryService;
    private readonly ISerializationService _serializationService;
    private readonly ICastingService _castingService;

    public TensorOperationServiceProvider()
    {
        _reshapeService = new ReshapeService();
        _deviceGuardService = new DeviceGuardService();
        _reductionService = new ReductionService();
        _linqService = new LinqService();
        _trigonometryService = new TrigonometryService();
        _serializationService = new SerializationService();
        _castingService = new CastingService();
        _powerService = new PowerService(this);
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

    public IPowerService GetPowerService()
    {
        return _powerService;
    }

    public IReductionService GetReductionService()
    {
        return _reductionService;
    }

    public ILinqService GetLinqService()
    {
        return _linqService;
    }

    public ITrigonometryService GetTrigonometryService()
    {
        return _trigonometryService;
    }

    public ISerializationService GetSerializationService()
    {
        return _serializationService;
    }

    public ICastingService GetCastingService()
    {
        return _castingService;
    }
}