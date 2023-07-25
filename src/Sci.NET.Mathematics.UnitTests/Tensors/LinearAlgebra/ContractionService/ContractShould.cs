// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Mathematics.Tensors.Manipulation;

namespace Sci.NET.Mathematics.UnitTests.Tensors.LinearAlgebra.ContractionService;

public class ContractShould
{
    private readonly Mock<IDeviceGuardService> _deviceGuardServiceMock;
    private readonly Mock<IPermutationService> _permutationServiceMock;
    private readonly Mock<IReshapeService> _reshapeServiceMock;
    private readonly Mock<IMatrixMultiplicationService> _matrixMultiplicationServiceMock;
    private readonly IContractionService _sut;

    public ContractShould()
    {
        _deviceGuardServiceMock = new Mock<IDeviceGuardService>();
        _permutationServiceMock = new Mock<IPermutationService>();
        _reshapeServiceMock = new Mock<IReshapeService>();
        _matrixMultiplicationServiceMock = new Mock<IMatrixMultiplicationService>();

        var tensorOperationServiceFactoryMock = new Mock<ITensorOperationServiceProvider>();

        tensorOperationServiceFactoryMock.Setup(x => x.GetDeviceGuardService())
            .Returns(_deviceGuardServiceMock.Object);

        tensorOperationServiceFactoryMock.Setup(x => x.GetPermutationService())
            .Returns(_permutationServiceMock.Object);

        tensorOperationServiceFactoryMock.Setup(x => x.GetReshapeService())
            .Returns(_reshapeServiceMock.Object);

        tensorOperationServiceFactoryMock.Setup(x => x.GetMatrixMultiplicationService())
            .Returns(_matrixMultiplicationServiceMock.Object);

        _sut = new Mathematics.Tensors.LinearAlgebra.Implementations.ContractionService(tensorOperationServiceFactoryMock.Object);
    }

    [Fact]
    public void CallAppropriateServices_GivenValidData()
    {
        // Arrange
        var leftMock = new Mock<ITensor<int>>();
        var rightMock = new Mock<ITensor<int>>();
        var permutedLeftMock = new Mock<ITensor<int>>();
        var permutedRightMock = new Mock<ITensor<int>>();
        var reshapedLeftMock = new Mock<ITensor<int>>();
        var reshapedRightMock = new Mock<ITensor<int>>();
        var reshapedLeftMatrixMock = new Mock<Matrix<int>>(10, 10, null);
        var reshapedRightMatrixMock = new Mock<Matrix<int>>(10, 10, null);
        var matrixMultipliedMock = new Mock<Matrix<int>>(10, 10, null);
        var resultMock = new Mock<ITensor<int>>();
        var leftShape = new Mathematics.Tensors.Shape(2, 3, 3);
        var rightShape = new Mathematics.Tensors.Shape(4, 3, 2);
        var resultShape = new Mathematics.Tensors.Shape(3, 4);
        var leftIndices = new int[] { 0, 2 };
        var rightIndices = new int[] { 2, 1 };
        var leftPermutation = new int[] { 1, 0, 2 };
        var rightPermutation = new int[] { 2, 1, 0 };

        leftMock.SetupGet(x => x.Shape)
            .Returns(leftShape);

        rightMock.SetupGet(x => x.Shape)
            .Returns(rightShape);

        _permutationServiceMock.Setup(x => x.Permute(leftMock.Object, It.IsAny<int[]>()))
            .Returns(permutedLeftMock.Object);

        _permutationServiceMock.Setup(x => x.Permute(rightMock.Object, It.IsAny<int[]>()))
            .Returns(permutedRightMock.Object);

        _reshapeServiceMock.Setup(x => x.Reshape(permutedLeftMock.Object, It.IsAny<Mathematics.Tensors.Shape>()))
            .Returns(reshapedLeftMock.Object);

        _reshapeServiceMock.Setup(x => x.Reshape(permutedRightMock.Object, It.IsAny<Mathematics.Tensors.Shape>()))
            .Returns(reshapedRightMock.Object);

        reshapedLeftMock.Setup(x => x.ToMatrix())
            .Returns(reshapedLeftMatrixMock.Object);

        reshapedRightMock.Setup(x => x.ToMatrix())
            .Returns(reshapedRightMatrixMock.Object);

        _matrixMultiplicationServiceMock.Setup(x => x.MatrixMultiply(It.IsAny<Matrix<int>>(), It.IsAny<Matrix<int>>()))
            .Returns(matrixMultipliedMock.Object);

        _reshapeServiceMock.Setup(x => x.Reshape(matrixMultipliedMock.Object, It.IsAny<Mathematics.Tensors.Shape>()))
            .Returns(resultMock.Object);

        // Act
        var result = _sut.Contract(leftMock.Object, rightMock.Object, leftIndices, rightIndices);

        // Assert
        result.Should().Be(resultMock.Object);

        _deviceGuardServiceMock.Verify(
            x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device),
            Times.Once);

        _permutationServiceMock.Verify(
            x => x.Permute(leftMock.Object, leftPermutation),
            Times.Once);

        _permutationServiceMock.Verify(
            x => x.Permute(rightMock.Object, rightPermutation),
            Times.Once);

        _reshapeServiceMock.Verify(
            x => x.Reshape(permutedLeftMock.Object, new Mathematics.Tensors.Shape(3, 6)),
            Times.Once);

        _reshapeServiceMock.Verify(
            x => x.Reshape(permutedRightMock.Object, new Mathematics.Tensors.Shape(6, 4)),
            Times.Once);

        reshapedLeftMock.Verify(x => x.ToMatrix(), Times.Once);
        reshapedRightMock.Verify(x => x.ToMatrix(), Times.Once);

        _matrixMultiplicationServiceMock.Verify(
            x => x.MatrixMultiply(reshapedLeftMatrixMock.Object, reshapedRightMatrixMock.Object),
            Times.Once);

        _reshapeServiceMock.Verify(
            x => x.Reshape(matrixMultipliedMock.Object, resultShape),
            Times.Once);
    }

    [Fact]
    public void ThrowException_WhenDevicesAreInvalid()
    {
        // Arrange
        var leftMock = new Mock<ITensor<int>>();
        var rightMock = new Mock<ITensor<int>>();

        leftMock.SetupGet(x => x.Device)
            .Returns(new Mock<IDevice>().Object);

        rightMock.SetupGet(x => x.Device)
            .Returns(new Mock<IDevice>().Object);

        _deviceGuardServiceMock.Setup(x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device))
            .Throws(new TensorDataLocalityException("Reason"));

        // Act
        var act = () => _sut.Contract(leftMock.Object, rightMock.Object, new int[] { 0 }, new int[] { 0 });

        // Assert
        act.Should().Throw<TensorDataLocalityException>();

        _deviceGuardServiceMock.Verify(
            x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device),
            Times.Once);

        _permutationServiceMock.VerifyNoOtherCalls();
        _reshapeServiceMock.VerifyNoOtherCalls();
        _matrixMultiplicationServiceMock.VerifyNoOtherCalls();
    }

    [Fact]
    public void ThrowArgumentException_WhenIndicesAreNotEqualLengths()
    {
        // Arrange
        var leftMock = new Mock<ITensor<int>>();
        var rightMock = new Mock<ITensor<int>>();
        var leftIndices = new int[] { 0 };
        var rightIndices = new int[] { 0, 1 };

        _deviceGuardServiceMock.Setup(x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device));

        // Act
        var act = () => _sut.Contract(leftMock.Object, rightMock.Object, leftIndices, rightIndices);

        // Assert
        act.Should().Throw<ArgumentException>();

        _deviceGuardServiceMock.Verify(
            x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device),
            Times.Once);

        _permutationServiceMock.VerifyNoOtherCalls();
        _reshapeServiceMock.VerifyNoOtherCalls();
        _matrixMultiplicationServiceMock.VerifyNoOtherCalls();
    }

    [Fact]
    public void ThrowArgumentException_WhenIndicesAreNotValid()
    {
        // Arrange
        var leftMock = new Mock<ITensor<int>>();
        var rightMock = new Mock<ITensor<int>>();
        var leftShape = new Mathematics.Tensors.Shape(2, 3);
        var rightShape = new Mathematics.Tensors.Shape(5, 6);

        var leftIndices = new int[] { 0 };
        var rightIndices = new int[] { 1 };

        leftMock.SetupGet(x => x.Shape)
            .Returns(leftShape);

        rightMock.SetupGet(x => x.Shape)
            .Returns(rightShape);

        _deviceGuardServiceMock.Setup(x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device));

        // Act
        var act = () => _sut.Contract(leftMock.Object, rightMock.Object, leftIndices, rightIndices);

        // Assert
        act.Should().Throw<ArgumentException>();

        _deviceGuardServiceMock.Verify(
            x => x.GuardBinaryOperation(leftMock.Object.Device, rightMock.Object.Device),
            Times.Once);

        _permutationServiceMock.VerifyNoOtherCalls();
        _reshapeServiceMock.VerifyNoOtherCalls();
        _matrixMultiplicationServiceMock.VerifyNoOtherCalls();
    }
}