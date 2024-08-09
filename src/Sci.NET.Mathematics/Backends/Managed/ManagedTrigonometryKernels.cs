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
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Sin(tensorBlock[i]));
    }

    public void Cos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Cos(tensorBlock[i]));
    }

    public void Tan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Tan(tensorBlock[i]));
    }

    public void Sin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sin = TNumber.Sin(tensorBlock[i]);
                resultBlock[i] = sin * sin;
            });
    }

    public void Sin2BackwardsInPlace<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result, long elementCount)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var multiplier = TNumber.CreateChecked(2);

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var input = tensorBlock[i];
                var (sin, cos) = TNumber.SinCos(input);

                resultBlock[i] = sin * cos * multiplier;
            });
    }

    public void Cos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cos = TNumber.Cos(tensorBlock[i]);
                resultBlock[i] = cos * cos;
            });
    }

    public void Tan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tan = TNumber.Tan(tensorBlock[i]);
                resultBlock[i] = tan * tan;
            });
    }

    public void Sinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Sinh(tensorBlock[i]));
    }

    public void Cosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Cosh(tensorBlock[i]));
    }

    public void Tanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Tanh(tensorBlock[i]));
    }

    public void Sinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sinh = TNumber.Sinh(tensorBlock[i]);
                resultBlock[i] = sinh * sinh;
            });
    }

    public void Cosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cosh = TNumber.Cosh(tensorBlock[i]);
                resultBlock[i] = cosh * cosh;
            });
    }

    public void Tanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var tanh = TNumber.Tanh(tensorBlock[i]);
                resultBlock[i] = tanh * tanh;
            });
    }

    public void Asin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Asin(tensorBlock[i]));
    }

    public void Acos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Acos(tensorBlock[i]));
    }

    public void Atan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Atan(tensorBlock[i]));
    }

    public void Asin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asin = TNumber.Asin(tensorBlock[i]);
                resultBlock[i] = asin * asin;
            });
    }

    public void Acos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acos = TNumber.Acos(tensorBlock[i]);
                resultBlock[i] = acos * acos;
            });
    }

    public void Atan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var atan = TNumber.Atan(tensorBlock[i]);
                resultBlock[i] = atan * atan;
            });
    }

    public void Asinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Asinh(tensorBlock[i]));
    }

    public void Acosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Acosh(tensorBlock[i]));
    }

    public void Atanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Atanh(tensorBlock[i]));
    }

    public void Asinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asinh = TNumber.Asinh(tensorBlock[i]);
                resultBlock[i] = asinh * asinh;
            });
    }

    public void Acosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acosh = TNumber.Acosh(tensorBlock[i]);
                resultBlock[i] = acosh * acosh;
            });
    }

    public void Atanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var atanh = TNumber.Atanh(tensorBlock[i]);
                resultBlock[i] = atanh * atanh;
            });
    }

    public void Csc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.One / TNumber.Sin(tensorBlock[i]));
    }

    public void Sec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.One / TNumber.Cos(tensorBlock[i]));
    }

    public void Cot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.One / TNumber.Tan(tensorBlock[i]));
    }

    public void Csc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var csc = TNumber.One / TNumber.Sin(tensorBlock[i]);
                resultBlock[i] = csc * csc;
            });
    }

    public void Sec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sec = TNumber.One / TNumber.Cos(tensorBlock[i]);
                resultBlock[i] = sec * sec;
            });
    }

    public void Cot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var cot = TNumber.One / TNumber.Tan(tensorBlock[i]);
                resultBlock[i] = cot * cot;
            });
    }

    public void Csch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.One / TNumber.Sinh(tensorBlock[i]));
    }

    public void Sech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.One / TNumber.Cosh(tensorBlock[i]));
    }

    public void Coth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.One / TNumber.Tanh(tensorBlock[i]));
    }

    public void Csch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var csch = TNumber.One / TNumber.Sinh(tensorBlock[i]);
                resultBlock[i] = csch * csch;
            });
    }

    public void Sech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sech = TNumber.One / TNumber.Cosh(tensorBlock[i]);
                resultBlock[i] = sech * sech;
            });
    }

    public void Coth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var coth = TNumber.One / TNumber.Tanh(tensorBlock[i]);
                resultBlock[i] = coth * coth;
            });
    }

    public void Acsc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Asin(TNumber.One / tensorBlock[i]));
    }

    public void Asec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Acos(TNumber.One / tensorBlock[i]));
    }

    public void Acot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Atan(TNumber.One / tensorBlock[i]));
    }

    public void Acsc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acsc = TNumber.Asin(TNumber.One / tensorBlock[i]);
                resultBlock[i] = acsc * acsc;
            });
    }

    public void Asec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asec = TNumber.Acos(TNumber.One / tensorBlock[i]);
                resultBlock[i] = asec * asec;
            });
    }

    public void Acot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acot = TNumber.Atan(TNumber.One / tensorBlock[i]);
                resultBlock[i] = acot * acot;
            });
    }

    public void Acsch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Asinh(TNumber.One / tensorBlock[i]));
    }

    public void Asech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Acosh(TNumber.One / tensorBlock[i]));
    }

    public void Acoth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Atanh(TNumber.One / tensorBlock[i]));
    }

    public void Acsch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acsch = TNumber.Asinh(TNumber.One / tensorBlock[i]);
                resultBlock[i] = acsch * acsch;
            });
    }

    public void Asech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var asech = TNumber.Acosh(TNumber.One / tensorBlock[i]);
                resultBlock[i] = asech * asech;
            });
    }

    public void Acoth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var acoth = TNumber.Atanh(TNumber.One / tensorBlock[i]);
                resultBlock[i] = acoth * acoth;
            });
    }
}