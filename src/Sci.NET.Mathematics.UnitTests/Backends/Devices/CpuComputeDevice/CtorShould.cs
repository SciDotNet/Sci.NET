// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.UnitTests.Backends.Devices.CpuComputeDevice;

public class CtorShould
{
    [Fact]
    public void SetAllPropertiesCorrectly()
    {
        // Arrange & Act
        var sut = new Mathematics.Backends.Devices.CpuComputeDevice();

        // Assert
        sut.Category.Should().Be(DeviceCategory.Cpu);
        sut.Name.Should().NotBeNullOrEmpty();
        sut.Id.Should().NotBe(Guid.Empty);
    }
}