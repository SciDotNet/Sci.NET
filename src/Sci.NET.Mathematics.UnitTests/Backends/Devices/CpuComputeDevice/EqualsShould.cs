// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.UnitTests.Backends.Devices.CpuComputeDevice;

public class EqualsShould
{
    [Fact]
    public void ReturnTrue_WhenIdsAreEqual()
    {
        // Arrange
        var sut = new Mathematics.Backends.Devices.CpuComputeDevice();
        var other = new Mathematics.Backends.Devices.CpuComputeDevice();

        // Act
        var result = sut.Equals(other);

        // Assert
        result.Should().BeTrue();
    }

    [Fact]
    [SuppressMessage("Maintainability", "CA1508:Avoid dead conditional code", Justification = "Test method")]
    public void ReturnFalse_WhenOtherIsNull()
    {
        // Arrange
        var sut = new Mathematics.Backends.Devices.CpuComputeDevice();
        var other = default(IDevice);

        // Act
        var result = sut.Equals(other);

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_WhenOtherIsNotCpuComputeDevice()
    {
        // Arrange
        var sut = new Mathematics.Backends.Devices.CpuComputeDevice();
        var other = new Mock<IDevice>();

        other.Setup(x => x.Id)
            .Returns(Guid.NewGuid());

        other.Setup(x => x.Name)
            .Returns("Some other device");

        // Act
        var result = sut.Equals(other.Object);

        // Assert
        result.Should().BeFalse();
    }
}