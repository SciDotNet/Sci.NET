// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sci.NET.Mathematics.Tensors.Common.Implementations;

internal class GradientAppenderService : IGradientAppenderService
{
    public void AddGradientIfRequired<TNumber>(
        ref ITensor<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        if (overrideRequiresGradient ?? (left.RequiresGradient || right.RequiresGradient))
        {
            result = result.WithGradient();
        }

        if (overrideRequiresGradient ?? left.RequiresGradient)
        {
            result.AddParent(
                name,
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
                name,
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
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction,
            name);

        result = resultITensor.ToScalar();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Vector<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction,
            name);

        result = resultITensor.ToVector();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Matrix<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultIt = result;

        AddGradientIfRequired(
            ref resultIt,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction,
            name);

        result = resultIt.ToMatrix();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Tensor<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            left,
            right,
            overrideRequiresGradient,
            leftGradientFunction,
            rightGradientFunction,
            name);

        result = resultITensor.ToTensor();
    }

    public void AddGradientIfRequired<TNumber>(
        ref ITensor<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        if (overrideRequiresGradient ?? input.RequiresGradient)
        {
            result = result.WithGradient();

            result.AddParent(
                name,
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
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction,
            name);

        result = resultITensor.ToScalar();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Vector<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction,
            name);

        result = resultITensor.ToVector();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Matrix<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction,
            name);

        result = resultITensor.ToMatrix();
    }

    public void AddGradientIfRequired<TNumber>(
        ref Tensor<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        name ??= "UnknownOperation";

        if (!name.EndsWith("Backward", StringComparison.InvariantCulture))
        {
            name = $"{name}Backward";
        }

        ITensor<TNumber> resultITensor = result;

        AddGradientIfRequired(
            ref resultITensor,
            input,
            overrideRequiresGradient,
            gradientFunction,
            name);

        result = resultITensor.ToTensor();
    }
}