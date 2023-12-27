// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Attributes;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// A test kernel.
/// </summary>
[PublicAPI]
public class TestKernel
{
    /// <summary>
    /// A test kernel method.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result.</param>
    [Kernel]
#pragma warning disable CA1822
    public void TestKernelMethod(IMemoryBlock<float> left, IMemoryBlock<float> right, IMemoryBlock<float> result)
#pragma warning restore CA1822
    {
        for (var i = 0; i < left.Length; i++)
        {
            result[i] = left[i] + right[i];
        }
    }
}