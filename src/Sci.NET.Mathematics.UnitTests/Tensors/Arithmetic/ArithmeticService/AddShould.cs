// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Pointwise;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Arithmetic.ArithmeticService;

public class AddShould
{
    private readonly Mock<IDeviceGuardService> _guardServiceMock;
    private readonly Mock<ITensorStorageKernels> _storageMock;
    private readonly Mock<IDevice> _deviceMock;
    private readonly Mock<ITensorBackend> _backendMock;
    private readonly IArithmeticService _sut;

    public AddShould()
    {
        var factoryMock = new Mock<ITensorOperationServiceProvider>();
        _guardServiceMock = new Mock<IDeviceGuardService>();
        factoryMock.Setup(f => f.GetDeviceGuardService()).Returns(_guardServiceMock.Object);

        _storageMock = new Mock<ITensorStorageKernels>();
        _backendMock = new Mock<ITensorBackend>();
        _deviceMock = new Mock<IDevice>();
        _sut = new Mathematics.Tensors.Pointwise.Implementations.ArithmeticService(factoryMock.Object);

        _backendMock.SetupGet(x => x.Device).Returns(_deviceMock.Object);
        _backendMock.SetupGet(x => x.Storage).Returns(new Mock<ITensorStorageKernels>().Object);

        _storageMock.Setup(x => x.Allocate<int>(It.IsAny<Mathematics.Tensors.Shape>()))
            .Returns(new Mock<IMemoryBlock<int>>().Object);
    }

    [Fact]
    public void ReturnScalar_GivenScalarAndScalar()
    {
        // Arrange
        var leftMock = new Mock<Scalar<int>>(_backendMock.Object);
        var rightMock = new Mock<Scalar<int>>(_backendMock.Object);

        _guardServiceMock.Setup(x => x.GuardBinaryOperation(It.IsAny<IDevice>(), It.IsAny<IDevice>()))
            .Verifiable();

        _backendMock.Setup(
                x => x.Arithmetic.Add(It.IsAny<Scalar<int>>(), It.IsAny<Scalar<int>>(), It.IsAny<Scalar<int>>()))
            .Verifiable();

        // Act
        var result = _sut.Add(leftMock.Object, rightMock.Object);

        // Assert
        result.Should().BeOfType<Scalar<int>>();

        _guardServiceMock.Verify(x => x.GuardBinaryOperation(_deviceMock.Object, _deviceMock.Object), Times.Once);

        _backendMock.Verify(
            x => x.Arithmetic.Add(leftMock.Object, rightMock.Object, It.IsAny<Scalar<int>>()),
            Times.Once);

        _backendMock.Verify(x => x.Storage.Allocate<int>(Mathematics.Tensors.Shape.Scalar), Times.Exactly(3));
        _backendMock.Verify(x => x.Device, Times.Exactly(2));
        _guardServiceMock.VerifyNoOtherCalls();
        _backendMock.VerifyNoOtherCalls();
        _storageMock.VerifyNoOtherCalls();
    }
}