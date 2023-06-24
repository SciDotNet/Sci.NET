// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Utils;

public class TensorMockFactory
{
    private readonly Mock<IDevice> _deviceMock;
    private readonly Mock<ITensorBackend> _backendMock;
    private readonly Mock<ITensorStorageKernels> _storageMock;

    public TensorMockFactory(
        Mock<ITensorBackend>? backendMock = null,
        Mock<ITensorStorageKernels>? storageMock = null,
        Mock<IDevice>? deviceMock = null)
    {
        _deviceMock = deviceMock ?? new Mock<IDevice>();
        _backendMock = backendMock ?? new Mock<ITensorBackend>();
        _storageMock = storageMock ?? new Mock<ITensorStorageKernels>();

        _backendMock.SetupGet(x => x.Device).Returns(_deviceMock.Object);
        _backendMock.SetupGet(x => x.Storage).Returns(_storageMock.Object);
    }

    public Mock<Scalar<TNumber>> CreateScalar<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return new Mock<Scalar<TNumber>>(_backendMock.Object);
    }
}