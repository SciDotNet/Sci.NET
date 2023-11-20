// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public interface IArithmeticOperatorTests
{
    public void ReturnExpectedResult_GivenScalarScalar(IDevice device);

    public void ReturnExpectedResult_GivenScalarVector(IDevice device);

    public void ReturnExpectedResult_GivenScalarMatrix(IDevice device);

    public void ReturnExpectedResult_GivenScalarTensor(IDevice device);

    public void ReturnExpectedResult_GivenVectorScalar(IDevice device);

    public void ReturnExpectedResult_GivenVectorVector(IDevice device);

    public void ReturnExpectedResult_GivenVectorMatrix(IDevice device);

    public void ReturnExpectedResult_GivenVectorTensor(IDevice device);

    public void ReturnExpectedResult_GivenMatrixScalar(IDevice device);

    public void ReturnExpectedResult_GivenMatrixVector(IDevice device);

    public void ReturnExpectedResult_GivenMatrixMatrix(IDevice device);

    public void ReturnExpectedResult_GivenMatrixTensor(IDevice device);

    public void ReturnExpectedResult_GivenTensorScalar(IDevice device);

    public void ReturnExpectedResult_GivenTensorVector(IDevice device);

    public void ReturnExpectedResult_GivenTensorMatrix(IDevice device);

    public void ReturnExpectedResult_GivenTensorTensor(IDevice device);
}