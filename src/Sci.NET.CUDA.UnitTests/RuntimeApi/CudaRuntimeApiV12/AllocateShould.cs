// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Runtime.Exceptions;
using Sci.NET.CUDA.Runtime.Structs;

namespace Sci.NET.CUDA.UnitTests.RuntimeApi.CudaRuntimeApiV12;

public class AllocateShould
{
    private readonly Runtime.CudaRuntimeApiV12 _sut;

    public AllocateShould()
    {
        _sut = new Runtime.CudaRuntimeApiV12();
    }

    [Fact]
    public unsafe void Succeed_GivenValidData()
    {
        // Arrange
        const long count = 1;

        // Act
        var result = _sut.Allocate<int>(count);

        // Assert
        ((long)result).Should().NotBe(IntPtr.Zero);
    }

    [Fact]
    public unsafe void Throw_GivenInvalidData()
    {
        // Arrange
        const long count = -1;

        // Act
        Action act = () => _sut.Allocate<int>(count);

        // Assert
        act.Should().Throw<CudaRuntimeException>()
            .Where(x => x.StatusCode == CudaStatusCode.CudaErrorMemoryAllocation);
    }
}