// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

#pragma warning disable CA1515
public interface IArithmeticTests
#pragma warning restore CA1515
{
    public void ReturnExpectedResult_GivenScalarsAndScalar(IDevice device);

    public void ReturnExpectedResult_GivenScalarAndVector(IDevice device);

    public void ReturnExpectedResult_GivenScalarAndMatrix(IDevice device);

    public void ReturnExpectedResult_GivenScalarTensor(IDevice device);

    public void ReturnExpectedResult_GivenVectorAndScalar(IDevice device);

    public void ReturnExpectedResult_GivenVectorAndVector(IDevice device);

    public void ReturnExpectedResult_GivenVectorAndMatrix(IDevice device);

    public void ReturnExpectedResult_GivenVectorAndTensor(IDevice device);

    public void ReturnExpectedResult_GivenMatrixAndScalar(IDevice device);

    public void ReturnExpectedResult_GivenMatrixAndVector(IDevice device);

    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device);

    public void ReturnExpectedResult_GivenMatrixAndTensor(IDevice device);

    public void ReturnExpectedResult_GivenTensorAndScalar(IDevice device);

    public void ReturnExpectedResult_GivenTensorAndVector(IDevice device);

    public void ReturnExpectedResult_GivenTensorAndMatrix(IDevice device);

    public void ReturnExpectedResult_GivenTensorAndTensor(IDevice device);
}