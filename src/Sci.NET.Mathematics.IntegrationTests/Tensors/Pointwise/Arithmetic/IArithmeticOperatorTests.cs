// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Pointwise.Arithmetic;

public interface IArithmeticOperatorTests
{
    public void ReturnExpectedResult_GivenScalarScalar();

    public void ReturnExpectedResult_GivenScalarVector();

    public void ReturnExpectedResult_GivenScalarMatrix();

    public void ReturnExpectedResult_GivenScalarTensor();

    public void ReturnExpectedResult_GivenVectorScalar();

    public void ReturnExpectedResult_GivenVectorVector();

    public void ReturnExpectedResult_GivenVectorMatrix();

    public void ReturnExpectedResult_GivenVectorTensor();

    public void ReturnExpectedResult_GivenMatrixScalar();

    public void ReturnExpectedResult_GivenMatrixVector();

    public void ReturnExpectedResult_GivenMatrixMatrix();

    public void ReturnExpectedResult_GivenMatrixTensor();

    public void ReturnExpectedResult_GivenTensorScalar();

    public void ReturnExpectedResult_GivenTensorVector();

    public void ReturnExpectedResult_GivenTensorMatrix();

    public void ReturnExpectedResult_GivenTensorTensor();
}