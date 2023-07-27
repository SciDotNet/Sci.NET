// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Optimizers;

/// <summary>
/// An Adam optimizer.
/// </summary>
/// <typeparam name="TNumber">The number type of the optimizer.</typeparam>
[PublicAPI]
public class Adam<TNumber> : IOptimizer<TNumber>, IDisposable
    where TNumber : unmanaged, INumber<TNumber>, IPowerFunctions<TNumber>, IRootFunctions<TNumber>
{
    private readonly Scalar<TNumber> _epsilon;
    private readonly Scalar<TNumber> _beta1;
    private readonly Scalar<TNumber> _beta2;
    private readonly Scalar<TNumber> _one;
    private readonly Scalar<TNumber> _zero;
    private readonly Scalar<TNumber> _minusOne;
    private readonly ITensor<TNumber>[] _m;
    private readonly ITensor<TNumber>[] _v;
    private Scalar<TNumber> _t;

    /// <summary>
    /// Initializes a new instance of the <see cref="Adam{TNumber}"/> class.
    /// </summary>
    /// <param name="parameterSet">The parameters to optimize.</param>
    /// <param name="learningRate">The learning rate for the optimizer.</param>
    /// <param name="beta1">The beta1 parameter for the optimizer.</param>
    /// <param name="beta2">The beta2 parameter for the optimizer.</param>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public Adam(
        ParameterCollection<TNumber> parameterSet,
        TNumber learningRate,
        TNumber beta1,
        TNumber beta2,
        IDevice? device = null)
    {
        Device = device ?? new CpuComputeDevice();
        LearningRate = new Scalar<TNumber>(learningRate, Device.GetTensorBackend());
        Parameters = parameterSet;
        _epsilon = new Scalar<TNumber>(GenericMath.Epsilon<TNumber>(), Device.GetTensorBackend());
        _beta1 = new Scalar<TNumber>(beta1, Device.GetTensorBackend());
        _beta2 = new Scalar<TNumber>(beta2, Device.GetTensorBackend());
        _t = new Scalar<TNumber>(TNumber.Zero, Device.GetTensorBackend());
        _m = new ITensor<TNumber>[Parameters.Sum(x => x.Count)];
        _v = new ITensor<TNumber>[Parameters.Sum(x => x.Count)];

        for (int i = 0; i < _m.Length; i++)
        {
            _m[i] = new Scalar<TNumber>();
            _v[i] = new Scalar<TNumber>();
        }

        _one = new Scalar<TNumber>(TNumber.One, Device.GetTensorBackend());
        _zero = new Scalar<TNumber>(TNumber.Zero, Device.GetTensorBackend());
        _minusOne = new Scalar<TNumber>(TNumber.Zero - TNumber.One, Device.GetTensorBackend());
    }

    /// <inheritdoc />
    public ParameterCollection<TNumber> Parameters { get; }

    /// <inheritdoc />
    public Scalar<TNumber> LearningRate { get; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public void Step()
    {
        _t = _t.Add(_one);

        var allParameters = Parameters.GetAll().ToList();

        _ = Parallel.For(
            0,
            allParameters.Count,
            i =>
            {
                var namedParameter = allParameters[i];

                using var weightedAvgM = _beta1.Multiply(_m[i]);
                using var weightedAvgV = _beta2.Multiply(_v[i]);
                using var oneMinusBeta1 = _one.Subtract(_beta1);
                using var oneMinusBeta2 = _one.Subtract(_beta2);
                using var squaredGradient = namedParameter.Gradient.Square();
                using var contributionM = oneMinusBeta1.Multiply(namedParameter.Gradient);
                using var contributionV = oneMinusBeta2.Multiply(squaredGradient);

                _m[i] = weightedAvgM.Add(contributionM);
                _v[i] = weightedAvgV.Add(contributionV);

                using var beta1PowT = _beta1.Pow(_t);
                using var beta2PowT = _beta2.Pow(_t);
                using var oneMinusBeta1PowT = _one.Subtract(beta1PowT);
                using var oneMinusBeta2PowT = _one.Subtract(beta2PowT);
                using var mHat = _m[i].ToTensor().Divide(oneMinusBeta1PowT);
                using var vHat = _v[i].ToTensor().Divide(oneMinusBeta2PowT);

                using var vHatSqrt = vHat.Sqrt();
                using var vHatSqrtAddEpsilon = vHatSqrt.Add(_epsilon);
                using var learningRateMHat = LearningRate.Multiply(mHat);
                using var lrMHatDivVHatSqrtAddEpsilon = learningRateMHat.Divide(vHatSqrtAddEpsilon);

                using var delta = _minusOne.Multiply(lrMHatDivVHatSqrtAddEpsilon);

                namedParameter.UpdateValue(delta);
            });
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        _epsilon.To<TDevice>();
        _beta1.To<TDevice>();
        _beta2.To<TDevice>();
        LearningRate.To<TDevice>();
        _t.To<TDevice>();
        _one.To<TDevice>();
        _zero.To<TDevice>();
        _minusOne.To<TDevice>();

        foreach (var namedParameter in Parameters.GetAll())
        {
            namedParameter.To<TDevice>();
        }

        foreach (var m in _m)
        {
            m.To<TDevice>();
        }

        foreach (var v in _v)
        {
            v.To<TDevice>();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the resources used by the optimizer.
    /// </summary>
    /// <param name="disposing">Whether the object is being disposed.</param>
    protected virtual void Dispose(bool disposing)
    {
        _epsilon.Dispose();
        _beta1.Dispose();
        _beta2.Dispose();
        _t.Dispose();
        _one.Dispose();
        _zero.Dispose();
        _minusOne.Dispose();
        LearningRate.Dispose();

        foreach (var m in _m)
        {
            m.Dispose();
        }

        foreach (var v in _v)
        {
            v.Dispose();
        }
    }
}