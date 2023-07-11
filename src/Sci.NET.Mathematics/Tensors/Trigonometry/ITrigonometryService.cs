// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Trigonometry;

/// <summary>
/// A service for performing trigonometric operations on <see cref="ITensor{TNumber}"/> instances.
/// </summary>
[PublicAPI]
public interface ITrigonometryService
{
    /// <summary>
    /// Calculates the sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The sine of the <see cref="Scalar{TNumber}"/>.</returns>
    public Scalar<TNumber> Sin<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the sine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The sine of the <see cref="Vector{TNumber}"/>.</returns>
    public Vector<TNumber> Sin<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The sine of the <see cref="Matrix{TNumber}"/>.</returns>
    public Matrix<TNumber> Sin<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The sine of the <see cref="Tensor{TNumber}"/>.</returns>
    public Tensor<TNumber> Sin<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Scalar{TNumber}"/>.</returns>
    public Scalar<TNumber> Cos<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Vector{TNumber}"/>.</returns>
    public Vector<TNumber> Cos<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Matrix{TNumber}"/>.</returns>
    public Matrix<TNumber> Cos<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Tensor{TNumber}"/>.</returns>
    public Tensor<TNumber> Cos<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the tan of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Scalar{TNumber}"/>.</returns>
    public Scalar<TNumber> Tan<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the tan of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Vector{TNumber}"/>.</returns>
    public Vector<TNumber> Tan<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the tan of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Matrix{TNumber}"/>.</returns>
    public Matrix<TNumber> Tan<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the tan of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Tensor{TNumber}"/>.</returns>
    public Tensor<TNumber> Tan<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the sinh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic sine of the <paramref name="scalar"/>.</returns>
    public Scalar<TNumber> Sinh<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the sinh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic sine of the <paramref name="vector"/>.</returns>
    public Vector<TNumber> Sinh<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the sinh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic sine of the <paramref name="matrix"/>.</returns>
    public Matrix<TNumber> Sinh<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the sinh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic sine of the <paramref name="tensor"/>.</returns>
    public Tensor<TNumber> Sinh<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the sinh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosine of the <paramref name="scalar"/>.</returns>
    public Scalar<TNumber> Cosh<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the cosh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosine of the <paramref name="vector"/>.</returns>
    public Vector<TNumber> Cosh<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the cosh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosine of the <paramref name="matrix"/>.</returns>
    public Matrix<TNumber> Cosh<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the cosh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosine of the <paramref name="tensor"/>.</returns>
    public Tensor<TNumber> Cosh<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic tangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the tanh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic tangent of the <paramref name="scalar"/>.</returns>
    public Scalar<TNumber> Tanh<TNumber>(Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic tangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the tanh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic tangent of the <paramref name="vector"/>.</returns>
    public Vector<TNumber> Tanh<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic tangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the tanh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic tangent of the <paramref name="matrix"/>.</returns>
    public Matrix<TNumber> Tanh<TNumber>(Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic tangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the tanh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic tangent of the <paramref name="tensor"/>.</returns>
    public Tensor<TNumber> Tanh<TNumber>(Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;
}