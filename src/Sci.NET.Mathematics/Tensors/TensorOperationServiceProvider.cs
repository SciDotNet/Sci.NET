// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Common.Implementations;
using Sci.NET.Mathematics.Tensors.Equality;
using Sci.NET.Mathematics.Tensors.Equality.Implementations;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Manipulation.Implementations;
using Sci.NET.Mathematics.Tensors.NeuralNetworks;
using Sci.NET.Mathematics.Tensors.NeuralNetworks.Implementations;
using Sci.NET.Mathematics.Tensors.Pointwise;
using Sci.NET.Mathematics.Tensors.Pointwise.Implementations;
using Sci.NET.Mathematics.Tensors.Random;
using Sci.NET.Mathematics.Tensors.Random.Implementations;
using Sci.NET.Mathematics.Tensors.Reduction;
using Sci.NET.Mathematics.Tensors.Reduction.Implementations;
using Sci.NET.Mathematics.Tensors.Serialization;
using Sci.NET.Mathematics.Tensors.Serialization.Implementations;
using Sci.NET.Mathematics.Tensors.Statistics;
using Sci.NET.Mathematics.Tensors.Statistics.Implementations;
using Sci.NET.Mathematics.Tensors.Trigonometry;
using Sci.NET.Mathematics.Tensors.Trigonometry.Implementations;

namespace Sci.NET.Mathematics.Tensors;

internal class TensorOperationServiceProvider : ITensorOperationServiceProvider
{
    private readonly Lazy<DeviceGuardService> _deviceGuardService;
    private readonly Lazy<PermutationService> _permutationService;
    private readonly Lazy<ReshapeService> _reshapeService;
    private readonly Lazy<ActivationFunctionService> _activationFunctionService;
    private readonly Lazy<BroadcastService> _broadcastService;
    private readonly Lazy<VectorOperationsService> _vectorOperationsService;
    private readonly Lazy<RandomService> _randomService;
    private readonly Lazy<NormalisationService> _normalisationService;
    private readonly Lazy<VarianceService> _varianceService;
    private readonly Lazy<GradientAppenderService> _gradientAppenderService;
    private readonly Lazy<TrigonometryService> _trigonometryService;
    private readonly Lazy<SerializationService> _serializationService;
    private readonly Lazy<CastingService> _castingService;
    private readonly Lazy<ContractionService> _contractionService;
    private readonly Lazy<ArithmeticService> _arithmeticService;
    private readonly Lazy<PowerService> _powerService;
    private readonly Lazy<ReductionService> _reductionService;
    private readonly Lazy<MatrixMultiplicationService> _matrixMultiplicationService;
    private readonly Lazy<ConvolutionService> _convolutionService;
    private readonly Lazy<ConcatenationService> _concatenationService;
    private readonly Lazy<TensorEqualityOperationService> _tensorEqualityOperationService;

    public TensorOperationServiceProvider()
    {
        _broadcastService = new Lazy<BroadcastService>(() => new BroadcastService());
        _reshapeService = new Lazy<ReshapeService>(() => new ReshapeService());
        _deviceGuardService = new Lazy<DeviceGuardService>(() => new DeviceGuardService());
        _vectorOperationsService = new Lazy<VectorOperationsService>(() => new VectorOperationsService());
        _randomService = new Lazy<RandomService>(() => new RandomService());
        _normalisationService = new Lazy<NormalisationService>(() => new NormalisationService());
        _varianceService = new Lazy<VarianceService>(() => new VarianceService());
        _gradientAppenderService = new Lazy<GradientAppenderService>(() => new GradientAppenderService());
        _trigonometryService = new Lazy<TrigonometryService>(() => new TrigonometryService());
        _serializationService = new Lazy<SerializationService>(() => new SerializationService());
        _castingService = new Lazy<CastingService>(() => new CastingService());
        _activationFunctionService = new Lazy<ActivationFunctionService>(() => new ActivationFunctionService());
        _permutationService = new Lazy<PermutationService>(() => new PermutationService());
        _arithmeticService = new Lazy<ArithmeticService>(() => new ArithmeticService());
        _contractionService = new Lazy<ContractionService>(() => new ContractionService());
        _convolutionService = new Lazy<ConvolutionService>(() => new ConvolutionService());
        _concatenationService = new Lazy<ConcatenationService>(() => new ConcatenationService());
        _tensorEqualityOperationService = new Lazy<TensorEqualityOperationService>(() => new TensorEqualityOperationService());
        _matrixMultiplicationService = new Lazy<MatrixMultiplicationService>(() => new MatrixMultiplicationService());
        _powerService = new Lazy<PowerService>(() => new PowerService());
        _reductionService = new Lazy<ReductionService>(() => new ReductionService());
    }

    public IMatrixMultiplicationService GetMatrixMultiplicationService()
    {
        return _matrixMultiplicationService.Value;
    }

    public IDeviceGuardService GetDeviceGuardService()
    {
        return _deviceGuardService.Value;
    }

    public IPermutationService GetPermutationService()
    {
        return _permutationService.Value;
    }

    public IReshapeService GetReshapeService()
    {
        return _reshapeService.Value;
    }

    public IContractionService GetContractionService()
    {
        return _contractionService.Value;
    }

    public IArithmeticService GetArithmeticService()
    {
        return _arithmeticService.Value;
    }

    public IPowerService GetPowerService()
    {
        return _powerService.Value;
    }

    public IReductionService GetReductionService()
    {
        return _reductionService.Value;
    }

    public ITrigonometryService GetTrigonometryService()
    {
        return _trigonometryService.Value;
    }

    public ISerializationService GetSerializationService()
    {
        return _serializationService.Value;
    }

    public ICastingService GetCastingService()
    {
        return _castingService.Value;
    }

    public IConvolutionService GetConvolutionService()
    {
        return _convolutionService.Value;
    }

    public IConcatenationService GetConcatenationService()
    {
        return _concatenationService.Value;
    }

    public IActivationFunctionService GetActivationFunctionService()
    {
        return _activationFunctionService.Value;
    }

    public IBroadcastService GetBroadcastingService()
    {
        return _broadcastService.Value;
    }

    public IVectorOperationsService GetVectorOperationsService()
    {
        return _vectorOperationsService.Value;
    }

    public IRandomService GetRandomService()
    {
        return _randomService.Value;
    }

    public INormalisationService GetNormalisationService()
    {
        return _normalisationService.Value;
    }

    public IVarianceService GetVarianceService()
    {
        return _varianceService.Value;
    }

    public IGradientAppenderService GetGradientAppenderService()
    {
        return _gradientAppenderService.Value;
    }

    public ITensorEqualityOperationService GetEqualityOperationService()
    {
        return _tensorEqualityOperationService.Value;
    }
}