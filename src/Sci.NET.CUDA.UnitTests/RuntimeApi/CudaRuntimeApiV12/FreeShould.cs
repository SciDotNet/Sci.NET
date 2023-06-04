// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.UnitTests.RuntimeApi.CudaRuntimeApiV12;

public class FreeShould
{
    private readonly Runtime.CudaRuntimeApiV12 _sut;

    public FreeShould()
    {
        _sut = new Runtime.CudaRuntimeApiV12();
    }

    [Fact]
    public unsafe void Succeed_GivenValidData()
    {
        // Arrange
        var data = _sut.Allocate<int>(1);

        // Act
        var act = () => _sut.Free(data);

        // Assert
        act.Should().NotThrow();
    }
}