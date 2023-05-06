﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed;

namespace Sci.NET.Mathematics.UnitTests.Backends.Managed.ManagedBackend;

public class CtorShould
{
    [Fact]
    public void Ctor_ShouldInitialisePropertiesCorrectly()
    {
        // Arrange
        var backend = new ManagedTensorBackend();

        // Assert
        backend.Arithmetic.Should().BeOfType<ManagedArithmeticBackend>();
        backend.LinearAlgebra.Should().BeOfType<ManagedLinearAlgebraBackend>();
        backend.Storage.Should().BeOfType<ManagedStorageBackend>();
        backend.Device.Should().BeOfType<CpuComputeDevice>();
    }
}