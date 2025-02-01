// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Common.Implementations;

internal class GradientAppenderService : IGradientAppenderService
{
    public void AddGradientIfRequired<TNumber>(
        ref ITensor<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (overrideRequiresGradient ?? (left.RequiresGradient || right.RequiresGradient))
        {
            result = result.WithGradient();
        }

        if (overrideRequiresGradient ?? left.RequiresGradient)
        {
            result.AddParent(
                left,
                grad =>
                {
                    var resultGrad = leftGradientFunction(grad);
                    return resultGrad.AsGradient();
                });
        }

        if (overrideRequiresGradient ?? right.RequiresGradient)
        {
            result.AddParent(
                right,
                grad =>
                {
                    var resultGrad = rightGradientFunction(grad);
                    return resultGrad.AsGradient();
                });
        }
    }

    public void AddGradientIfRequired<TNumber>(
        ref Scalar<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction);

        result = resultITensor.ToScalar();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Vector<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction);

        result = resultITensor.ToVector();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Matrix<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultIt = result;

        AddGradientIfRequired(
            ref resultIt,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction);

        result = resultIt.ToMatrix();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Tensor<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction);

        result = resultITensor.ToTensor();
    }

    public void AddGradientIfRequired<TNumber>(
        ref ITensor<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (overrideRequiresGradient ?? input.RequiresGradient)
        {
            result = result.WithGradient();

            result.AddParent(
                input,
                grad =>
                {
                    var resultGrad = gradientFunction(grad.AsGradient());
                    return resultGrad.AsGradient();
                });
        }
    }

    public void AddGradientIfRequired<TNumber>(
        ref Scalar<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction);

        result = resultITensor.ToScalar();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Vector<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction);

        result = resultITensor.ToVector();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Matrix<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction);

        result = resultITensor.ToMatrix();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Tensor<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction);

        result = resultITensor.ToTensor();
    }
}