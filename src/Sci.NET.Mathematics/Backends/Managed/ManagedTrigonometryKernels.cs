// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedTrigonometryKernels : ITrigonometryKernels
{
    public void Sin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Sin(input[i]));
    }

    public void Cos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Cos(input[i]));
    }

    public void Tan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Tan(input[i]));
    }

    public void Sin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sin = TNumber.Sin(input[i]);
                output[i] = sin * sin;
            });
    }

    public void Cos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cos = TNumber.Cos(input[i]);
                output[i] = cos * cos;
            });
    }

    public void Tan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tan = TNumber.Tan(input[i]);
                output[i] = tan * tan;
            });
    }

    public void Sinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Sinh(input[i]));
    }

    public void Cosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Cosh(input[i]));
    }

    public void Tanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Tanh(input[i]));
    }

    public void Sinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(input[i]);
                output[i] = sinh * sinh;
            });
    }

    public void Cosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cosh = TNumber.Cosh(input[i]);
                output[i] = cosh * cosh;
            });
    }

    public void Tanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tanh = TNumber.Tanh(input[i]);
                output[i] = tanh * tanh;
            });
    }

    public void Asin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Asin(input[i]));
    }

    public void Acos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = input[i] < -TNumber.One || TNumber.NaN > TNumber.One ? TNumber.NaN : TNumber.Acos(input[i]));
    }

    public void Atan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Atan(input[i]));
    }

    public void Asin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asin = TNumber.Asin(input[i]);
                output[i] = asin * asin;
            });
    }

    public void Acos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acos = TNumber.Acos(input[i]);
                output[i] = acos * acos;
            });
    }

    public void Atan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var atan = TNumber.Atan(input[i]);
                output[i] = atan * atan;
            });
    }

    public void Asinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Asinh(input[i]));
    }

    public void Acosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Acosh(input[i]));
    }

    public void Atanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Atanh(input[i]));
    }

    public void Asinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asinh = TNumber.Asinh(input[i]);
                output[i] = asinh * asinh;
            });
    }

    public void Acosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acosh = TNumber.Acosh(input[i]);
                output[i] = acosh * acosh;
            });
    }

    public void Atanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var atanh = TNumber.Atanh(input[i]);
                output[i] = atanh * atanh;
            });
    }

    public void Csc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.One / TNumber.Sin(input[i]));
    }

    public void Sec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.One / TNumber.Cos(input[i]));
    }

    public void Cot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                // TODO - Consider performance/accuracy of -tan(x+pi/2) vs cos(x)/sin(x), due to boxing/etc.
                var (sin, cos) = TNumber.SinCos(input[i]);

                output[i] = cos / sin;
            });
    }

    public void Csc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sin = TNumber.Sin(input[i]);
                output[i] = TNumber.One / (sin * sin);
            });
    }

    public void Sec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cos = TNumber.Cos(input[i]);
                output[i] = TNumber.One / (cos * cos);
            });
    }

    public void Cot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                // TODO - Consider performance/accuracy of -tan(x+pi/2) vs cos(x)/sin(x), due to boxing/etc.
                var (sin, cos) = TNumber.SinCos(input[i]);
                var cosOverSin = cos / sin;

                output[i] = cosOverSin * cosOverSin;
            });
    }

    public void Csch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.One / TNumber.Sinh(input[i]));
    }

    public void Sech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.One / TNumber.Cosh(input[i]));
    }

    public void Coth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                // TODO - Consider performance/accuracy of -tanh(x+pi/2) vs cosh(x)/sinh(x), due to boxing/etc.
                var sinh = TNumber.Sinh(input[i]);
                var cosh = TNumber.Cosh(input[i]);

                output[i] = cosh / sinh;
            });
    }

    public void Csch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(input[i]);
                output[i] = TNumber.One / (sinh * sinh);
            });
    }

    public void Sech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cosh = TNumber.Cosh(input[i]);
                output[i] = TNumber.One / (cosh * cosh);
            });
    }

    public void Coth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                // TODO - Consider performance/accuracy of -tanh(x+pi/2) vs cosh(x)/sinh(x), due to boxing/etc.
                var sinh = TNumber.Sinh(input[i]);
                var cosh = TNumber.Cosh(input[i]);
                var coshOverSinh = cosh / sinh;

                output[i] = coshOverSinh * coshOverSinh;
            });
    }

    public void Acsc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Asin(TNumber.One / input[i]));
    }

    public void Asec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Acos(TNumber.One / input[i]));
    }

    public void Acot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Atan(TNumber.One / input[i]));
    }

    public void Acsc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acsc = TNumber.Asin(TNumber.One / input[i]);
                output[i] = acsc * acsc;
            });
    }

    public void Asec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asec = TNumber.Acos(TNumber.One / input[i]);
                output[i] = asec * asec;
            });
    }

    public void Acot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acot = TNumber.Atan(TNumber.One / input[i]);
                output[i] = acot * acot;
            });
    }

    public void Acsch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Asinh(TNumber.One / input[i]));
    }

    public void Asech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Acosh(TNumber.One / input[i]));
    }

    public void Acoth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = TNumber.Atanh(TNumber.One / input[i]));
    }

    public void Acsch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acsch = TNumber.Asinh(TNumber.One / input[i]);
                output[i] = acsch * acsch;
            });
    }

    public void Asech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asech = TNumber.Acosh(TNumber.One / input[i]);
                output[i] = asech * asech;
            });
    }

    public void Acoth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acoth = TNumber.Atanh(TNumber.One / input[i]);
                output[i] = acoth * acoth;
            });
    }

    public void SinBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = grad[i] * TNumber.Cos(input[i]));
    }

    public void CosBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = grad[i] * -TNumber.Sin(input[i]));
    }

    public void TanBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cos = TNumber.Cos(input[i]);
                output[i] = grad[i] * (TNumber.One / (cos * cos));
            });
    }

    public void Sin2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var (sin, cos) = TNumber.SinCos(input[i]);
                var two = TNumber.One + TNumber.One;
                output[i] = grad[i] * two * sin * cos;
            });
    }

    public void Cos2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var (sin, cos) = TNumber.SinCos(input[i]);
                var two = TNumber.One + TNumber.One;
                output[i] = grad[i] * -two * sin * cos;
            });
    }

    public void Tan2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tan = TNumber.Tan(input[i]);
                var cos = TNumber.Cos(input[i]);
                var sec2 = TNumber.One / (cos * cos);
                var two = TNumber.One + TNumber.One;
                output[i] = grad[i] * two * tan * sec2;
            });
    }

    public void SinhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cosh = TNumber.Cosh(input[i]);
                output[i] = grad[i] * cosh;
            });
    }

    public void CoshBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(input[i]);
                output[i] = grad[i] * sinh;
            });
    }

    public void TanhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tanh = TNumber.Tanh(input[i]);
                output[i] = grad[i] * (TNumber.One - (tanh * tanh));
            });
    }

    public void Sinh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(input[i]);
                var cosh = TNumber.Cosh(input[i]);
                var two = TNumber.One + TNumber.One;
                output[i] = grad[i] * two * sinh * cosh;
            });
    }

    public void Cosh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(input[i]);
                var cosh = TNumber.Cosh(input[i]);
                var two = TNumber.One + TNumber.One;
                output[i] = grad[i] * two * sinh * cosh;
            });
    }

    public void Tanh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tanh = TNumber.Tanh(input[i]);
                var cosh = TNumber.Cosh(input[i]);
                var sech2 = TNumber.One / (cosh * cosh);
                var two = TNumber.One + TNumber.One;
                output[i] = grad[i] * two * tanh * sech2;
            });
    }

    public void AsinBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sqrt = TNumber.Sqrt(TNumber.One - (input[i] * input[i]));
                output[i] = grad[i] * (TNumber.One / sqrt);
            });
    }

    public void AcosBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sqrt = TNumber.Sqrt(TNumber.One - (input[i] * input[i]));
                output[i] = grad[i] * (-TNumber.One / sqrt);
            });
    }

    public void AtanBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var onePlusX2 = TNumber.One + (input[i] * input[i]);
                output[i] = grad[i] * (TNumber.One / onePlusX2);
            });
    }

    public void Asin2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var asin = TNumber.Asin(input[i]);
                var sqrt1MinusX2 = TNumber.Sqrt(TNumber.One - (input[i] * input[i]));

                output[i] = grad[i] * (two * asin / sqrt1MinusX2);
            });
    }

    public void Acos2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var acos = TNumber.Acos(input[i]);
                var sqrt1MinusX2 = TNumber.Sqrt(TNumber.One - (input[i] * input[i]));

                output[i] = grad[i] * (-two * acos / sqrt1MinusX2);
            });
    }

    public void Atan2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var atan = TNumber.Atan(input[i]);
                var twoAtan = two * atan;
                var x2Plus1 = (input[i] * input[i]) + TNumber.One;

                output[i] = grad[i] * (twoAtan / x2Plus1);
            });
    }

    public void AsinhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sqrt = TNumber.Sqrt(TNumber.One + (input[i] * input[i]));
                output[i] = grad[i] * (TNumber.One / sqrt);
            });
    }

    public void AcoshBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sqrt = TNumber.Sqrt((input[i] * input[i]) - TNumber.One);
                output[i] = grad[i] * (TNumber.One / sqrt);
            });
    }

    public void AtanhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var oneMinusX2 = TNumber.One - (input[i] * input[i]);
                output[i] = grad[i] * (TNumber.One / oneMinusX2);
            });
    }

    public void Asinh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var asinh = TNumber.Asinh(input[i]);
                var sqrtOnePlusX2 = TNumber.Sqrt(TNumber.One + (input[i] * input[i]));

                output[i] = grad[i] * (two * asinh / sqrtOnePlusX2);
            });
    }

    public void Acosh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var acosh = TNumber.Acosh(input[i]);
                var sqrtX2MinusOne = TNumber.Sqrt((input[i] * input[i]) - TNumber.One);

                output[i] = grad[i] * (two * acosh / sqrtX2MinusOne);
            });
    }

    public void Atanh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var atanh = TNumber.Atanh(input[i]);
                var oneMinusX2 = TNumber.One - (input[i] * input[i]);

                output[i] = grad[i] * (two * atanh / oneMinusX2);
            });
    }

    public void CscBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var (sin, cos) = TNumber.SinCos(input[i]);
                var csc = TNumber.One / sin;
                var cot = cos / sin;
                output[i] = grad[i] * -csc * cot;
            });
    }

    public void SecBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var (sin, cos) = TNumber.SinCos(input[i]);
                var sec = TNumber.One / cos;
                var tan = sin / cos;
                output[i] = grad[i] * sec * tan;
            });
    }

    public void CotBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sin = TNumber.Sin(input[i]);
                var csc2 = TNumber.One / (sin * sin);

                output[i] = grad[i] * -csc2;
            });
    }

    public void Csc2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var (sin, cos) = TNumber.SinCos(input[i]);
                var twoCosX = two * cos;
                var sin3 = sin * sin * sin;

                output[i] = grad[i] * -(twoCosX / sin3);
            });
    }

    public void Sec2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var cos = TNumber.Cos(input[i]);
                var sec2 = TNumber.One / (cos * cos);
                var tan = TNumber.Tan(input[i]);

                output[i] = grad[i] * two * sec2 * tan;
            });
    }

    public void Cot2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var (sin, cos) = TNumber.SinCos(input[i]);
                var cot = cos / sin;
                var csc2 = TNumber.One / (sin * sin);

                output[i] = grad[i] * -two * csc2 * cot;
            });
    }

    public void CschBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(input[i]);
                var csch = TNumber.One / sinh;
                var coth = TNumber.Cosh(input[i]) / sinh;

                output[i] = grad[i] * -csch * coth;
            });
    }

    public void SechBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cosh = TNumber.Cosh(input[i]);
                var sech = TNumber.One / cosh;
                var tanh = TNumber.Tanh(input[i]);

                output[i] = grad[i] * -sech * tanh;
            });
    }

    public void CothBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var csch = TNumber.One / TNumber.Sinh(input[i]);

                output[i] = grad[i] * -(csch * csch);
            });
    }

    public void Csch2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var sinh = TNumber.Sinh(input[i]);
                var csch2 = TNumber.One / (sinh * sinh);
                var coth = TNumber.Cosh(input[i]) / sinh;

                output[i] = grad[i] * -two * csch2 * coth;
            });
    }

    public void Sech2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var cosh = TNumber.Cosh(input[i]);
                var sech2 = TNumber.One / (cosh * cosh);
                var tanh = TNumber.Tanh(input[i]);

                output[i] = grad[i] * -two * sech2 * tanh;
            });
    }

    public void Coth2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var sinh = TNumber.Sinh(input[i]);
                var sinh2 = sinh * sinh;
                var cosh = TNumber.Cosh(input[i]);
                var coth = cosh / sinh;

                output[i] = grad[i] * -two * (TNumber.One / sinh2) * coth;
            });
    }

    public void AcscBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt(TNumber.One - (TNumber.One / x2));
                var denominator = -TNumber.One / (x2 * sqrt);
                output[i] = grad[i] * denominator;
            });
    }

    public void AsecBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt(TNumber.One - (TNumber.One / x2));
                var denominator = TNumber.One / (x2 * sqrt);
                output[i] = grad[i] * denominator;
            });
    }

    public void AcotBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = grad[i] * -TNumber.One / (TNumber.One + (input[i] * input[i])));
    }

    public void Acsc2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt(TNumber.One - (TNumber.One / x2));
                var denominator = x2 * sqrt;
                var acsc = TNumber.Asin(TNumber.One / input[i]);
                var derivative = -two * acsc / denominator;

                output[i] = grad[i] * derivative;
            });
    }

    public void Asec2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt(TNumber.One - (TNumber.One / x2));
                var denominator = x2 * sqrt;
                var acsc = TNumber.Acos(TNumber.One / input[i]);
                var derivative = two * acsc / denominator;

                output[i] = grad[i] * derivative;
            });
    }

    public void Acot2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var acot = TNumber.Atan(TNumber.One / input[i]);
                var x2 = input[i] * input[i];

                output[i] = grad[i] * (-two * acot / (x2 + TNumber.One));
            });
    }

    public void AcschBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt(TNumber.One + (TNumber.One / x2)) * x2;

                output[i] = grad[i] * (-TNumber.One / sqrt);
            });
    }

    public void AsechBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt((TNumber.One / x2) - TNumber.One) * x2;

                output[i] = grad[i] * (-TNumber.One / sqrt);
            });
    }

    public void AcothBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => output[i] = grad[i] * -TNumber.One / ((input[i] * input[i]) - TNumber.One));
    }

    public void Acsch2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt(TNumber.One + (TNumber.One / x2)) * x2;
                var acsch = TNumber.Asinh(TNumber.One / input[i]);

                output[i] = grad[i] * (-two * acsch / sqrt);
            });
    }

    public void Asech2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var x2 = input[i] * input[i];
                var sqrt = TNumber.Sqrt((TNumber.One / x2) - TNumber.One) * x2;
                var asech = TNumber.Acosh(TNumber.One / input[i]);

                output[i] = grad[i] * (-two * asech / sqrt);
            });
    }

    public void Acoth2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var input = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var output = (SystemMemoryBlock<TNumber>)result.Memory;
        var grad = (SystemMemoryBlock<TNumber>)gradient.Memory;

        _ = LazyParallelExecutor.For(
            0,
            input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var two = TNumber.One + TNumber.One;
                var acoth = TNumber.Atanh(TNumber.One / input[i]);
                var x2 = input[i] * input[i];
                var denominator = x2 - TNumber.One;

                output[i] = grad[i] * (-two * acoth / denominator);
            });
    }
}