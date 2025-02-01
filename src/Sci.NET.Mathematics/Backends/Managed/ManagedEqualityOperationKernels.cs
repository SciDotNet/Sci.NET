// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedEqualityOperationKernels : IEqualityOperationKernels
{
    public void PointwiseEqualsKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].Equals(rightBlock[i]) ? TNumber.One : TNumber.Zero;
        }
    }

    public void PointwiseNotEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].Equals(rightBlock[i]) ? TNumber.Zero : TNumber.One;
        }
    }

    public void PointwiseGreaterThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].CompareTo(rightBlock[i]) > 0 ? TNumber.One : TNumber.Zero;
        }
    }

    public void PointwiseGreaterThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].CompareTo(rightBlock[i]) >= 0 ? TNumber.One : TNumber.Zero;
        }
    }

    public void PointwiseLessThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].CompareTo(rightBlock[i]) < 0 ? TNumber.One : TNumber.Zero;
        }
    }

    public void PointwiseLessThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].CompareTo(rightBlock[i]) <= 0 ? TNumber.One : TNumber.Zero;
        }
    }
}