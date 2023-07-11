// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Trigonometry.Implementations;

internal class TrigonometryService : ITrigonometryService
{
    public Scalar<TNumber> Sin<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = scalar.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Trigonometry.Sin(scalar, result);

        return result;
    }

    public Vector<TNumber> Sin<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = vector.Backend;
        var result = new Vector<TNumber>(vector.Length, backend);

        backend.Trigonometry.Sin(vector, result);

        return result;
    }

    public Matrix<TNumber> Sin<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = matrix.Backend;
        var result = new Matrix<TNumber>(matrix.Rows, matrix.Columns, backend);

        backend.Trigonometry.Sin(matrix, result);

        return result;
    }

    public Tensor<TNumber> Sin<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Trigonometry.Sin(tensor, result);

        return result;
    }

    public Scalar<TNumber> Cos<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = scalar.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Trigonometry.Cos(scalar, result);

        return result;
    }

    public Vector<TNumber> Cos<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = vector.Backend;
        var result = new Vector<TNumber>(vector.Length, backend);

        backend.Trigonometry.Cos(vector, result);

        return result;
    }

    public Matrix<TNumber> Cos<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = matrix.Backend;
        var result = new Matrix<TNumber>(matrix.Rows, matrix.Columns, backend);

        backend.Trigonometry.Cos(matrix, result);

        return result;
    }

    public Tensor<TNumber> Cos<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Trigonometry.Cos(tensor, result);

        return result;
    }

    public Scalar<TNumber> Tan<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = scalar.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Trigonometry.Tan(scalar, result);

        return result;
    }

    public Vector<TNumber> Tan<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = vector.Backend;
        var result = new Vector<TNumber>(vector.Length, backend);

        backend.Trigonometry.Tan(vector, result);

        return result;
    }

    public Matrix<TNumber> Tan<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = matrix.Backend;
        var result = new Matrix<TNumber>(matrix.Rows, matrix.Columns, backend);

        backend.Trigonometry.Tan(matrix, result);

        return result;
    }

    public Tensor<TNumber> Tan<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Trigonometry.Tan(tensor, result);

        return result;
    }

    public Scalar<TNumber> Sinh<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = scalar.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Trigonometry.Sinh(scalar, result);

        return result;
    }

    public Vector<TNumber> Sinh<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = vector.Backend;
        var result = new Vector<TNumber>(vector.Length, backend);

        backend.Trigonometry.Sinh(vector, result);

        return result;
    }

    public Matrix<TNumber> Sinh<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = matrix.Backend;
        var result = new Matrix<TNumber>(matrix.Rows, matrix.Columns, backend);

        backend.Trigonometry.Sinh(matrix, result);

        return result;
    }

    public Tensor<TNumber> Sinh<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Trigonometry.Sinh(tensor, result);

        return result;
    }

    public Scalar<TNumber> Cosh<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = scalar.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Trigonometry.Cosh(scalar, result);

        return result;
    }

    public Vector<TNumber> Cosh<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = vector.Backend;
        var result = new Vector<TNumber>(vector.Length, backend);

        backend.Trigonometry.Cosh(vector, result);

        return result;
    }

    public Matrix<TNumber> Cosh<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = matrix.Backend;
        var result = new Matrix<TNumber>(matrix.Rows, matrix.Columns, backend);

        backend.Trigonometry.Cosh(matrix, result);

        return result;
    }

    public Tensor<TNumber> Cosh<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Trigonometry.Cosh(tensor, result);

        return result;
    }

    public Scalar<TNumber> Tanh<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = scalar.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Trigonometry.Tanh(scalar, result);

        return result;
    }

    public Vector<TNumber> Tanh<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = vector.Backend;
        var result = new Vector<TNumber>(vector.Length, backend);

        backend.Trigonometry.Tanh(vector, result);

        return result;
    }

    public Matrix<TNumber> Tanh<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = matrix.Backend;
        var result = new Matrix<TNumber>(matrix.Rows, matrix.Columns, backend);

        backend.Trigonometry.Tanh(matrix, result);

        return result;
    }

    public Tensor<TNumber> Tanh<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Trigonometry.Tanh(tensor, result);

        return result;
    }
}