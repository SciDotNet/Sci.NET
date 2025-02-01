// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for trigonometry kernels.
/// </summary>
[PublicAPI]
public interface ITrigonometryKernels
{
    /// <summary>
    /// Computes the element-wise sine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise tangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise sine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise tangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic sine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic cosine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic tangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic sine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic cosine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic tangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arctangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arctangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arcsine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arccosine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arctangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arcsine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arccosine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arctangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise secant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cotangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise secant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cotangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic cosecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic secant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic cotangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Coth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic cosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic secant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic cotangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Coth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acsc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccotangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acsc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccotangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arccosecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acsch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arcsecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arccotangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acoth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arccosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acsch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arcsecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise hyperbolic arccotangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acoth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the sine function for the given input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the sine function was originally evaluated.</param>
    /// <param name="gradient">The gradient of the output tensor with respect to a subsequent operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> representing the gradient of the sine function with respect to the input tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SinBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the cosine function with respect to its input for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is computed.</param>
    /// <param name="gradient">The gradient of the output with respect to the function's input.</param>
    /// <param name="result">The output gradient <see cref="ITensor{TNumber}"/> of the cosine function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CosBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the backwards operation of the tangent function element-wise
    /// for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> after applying the backwards operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void TanBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the sine function with respect to its input for a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the forward input to the sine function.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> that stores the computed gradients.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sin2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the cosine function with respect to its input for a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the forward input to the cosine function.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> that stores the computed gradients.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cos2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the backwards operation of the tangent function element-wise.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> after applying the backwards operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tan2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the hyperbolic sine function (sinh) during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the operation is differentiated.</param>
    /// <param name="gradient">The gradient propagated from the subsequent layer as a <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> after applying the backward differentiation of sinh.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SinhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the hyperbolic cosine (cosh) operation for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the values for which the backward operation is computed.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> containing the gradients passed from the next layer in a computational graph.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradients with respect to the input tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CoshBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the hyperbolic tangent function for a given input tensor during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is being computed.</param>
    /// <param name="gradient">The gradient tensor propagated from subsequent layers.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> storing the result of the Tanh gradient computation.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    public void TanhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise backward operation for the hyperbolic sine squared function of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the backward operation is computed.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the results of the backward operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sinh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise hyperbolic cosine squared operation for a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> to compute the gradient from.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the result of the backward computation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cosh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise derivative of the squared hyperbolic tangent (Tanh^2) of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tanh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arcsine function with respect to its input for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the arcsine gradient is computed.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient of the arcsine function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AsinBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arc cosine function for the given <see cref="ITensor{TNumber}"/> input.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the arc cosine gradient is calculated.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> storing the computed gradient values.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AcosBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arctangent function with respect to its input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> to compute the gradient for.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> storing the gradient of the arctangent function.</param>
    /// <typeparam name="TNumber">The numeric type of the elements in the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AtanBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise square of the arcsine function for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the arcsine square gradient is computed.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output of the forward operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> storing the computed gradient of the arcsine square function.</param>
    /// <typeparam name="TNumber">The numeric type of the elements in the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asin2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arc cosine squared function with respect to its input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the original argument of the arc cosine squared function.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient of the output with respect to the arc cosine squared function.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the gradient of the input with respect to the original argument.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acos2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient for the two-argument arc tangent operation with respect to the input tensor and gradient.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the original inputs to the arc tangent operation.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient of the preceding operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient of the arc tangent operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>, constrained to IEEE 754 compliant floating-point types.</typeparam>
    public void Atan2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the inverse hyperbolic sine function for a <see cref="ITensor{TNumber}"/> during the backward pass.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the original data.</param>
    /// <param name="gradient">The gradient tensor from the subsequent layer in the computational graph.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradients.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AsinhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the inverse hyperbolic cosine function with respect to its input.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is computed.</param>
    /// <param name="gradient">The gradient of the output function with respect to its result.</param>
    /// <param name="result">The resulting <see cref="ITensor{TNumber}"/> containing the computed gradient.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AcoshBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise inverse hyperbolic tangent function applied to a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradient">The gradient of the output with respect to the function's result.</param>
    /// <param name="result">The resulting gradient of the output with respect to the input <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AtanhBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise inverse hyperbolic sine operation for a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradient">The incoming gradient <see cref="ITensor{TNumber}"/> with respect to the output.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> representing the gradient with respect to the input.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asinh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise squared inverse hyperbolic cosine operation for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which gradients are computed.</param>
    /// <param name="gradient">The incoming gradient <see cref="ITensor{TNumber}"/> from the subsequent computation layer.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradients.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acosh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the backward pass for the element-wise inverse hyperbolic tangent operation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the backward computation is performed.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> from the subsequent layer or computation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradients.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atanh2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the cosecant function with respect to its input tensor during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> containing the original input to the cosecant function.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient of the loss with respect to the output of the cosecant function.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient of the loss with respect to the input tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CscBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the secant function with respect to its input tensor during the backward pass.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the tensor for which the backward computation is being performed.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient propagated from subsequent layers.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> where the computed gradient with respect to the input is stored.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SecBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the cotangent function with respect to the input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the backward pass of cotangent is calculated.</param>
    /// <param name="gradient">The gradient propagated from the subsequent operations.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the result of the cotangent backward computation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CotBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the backward gradient for the squared cosecant operation on a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> on which the Csc² operation was applied.</param>
    /// <param name="gradient">The gradient of the scalar value with respect to the output of the Csc² operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> where the computed gradient will be stored.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csc2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the secant squared operation with respect to its input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> on which the operation is defined.</param>
    /// <param name="gradient">The input gradient <see cref="ITensor{TNumber}"/> with respect to the function output.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient with respect to the input tensor.</param>
    /// <typeparam name="TNumber">The numeric type of the tensors.</typeparam>
    public void Sec2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise cotangent squared operation for the given input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the cotangent squared gradient is computed.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient from subsequent operations.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the resulting gradient.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cot2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise hyperbolic cosecant function with respect to its input.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the values for which the gradient is being calculated.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> of the output with respect to this operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient with respect to the input.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CschBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient for the hyperbolic secant function during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the original computation values.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> containing upstream gradients.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> storing the propagated gradients.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SechBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the hyperbolic cotangent function with respect to the input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> to compute the gradient for.</param>
    /// <param name="gradient">The gradient tensor provided as input.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> that will contain the computed result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CothBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the backwards gradient of the element-wise squared hyperbolic cosecant function.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is being computed.</param>
    /// <param name="gradient">The gradient of the loss function with respect to the output of the squared hyperbolic cosecant function.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient of the loss function with respect to the input tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csch2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the derivative of the squared hyperbolic secant function (sech^2) with respect to a given gradient tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the values for which the derivative of sech^2 is computed.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient to propagate.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> that stores the computed derivative results.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sech2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise hyperbolic cotangent squared function for a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the original values.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the gradient propagated from the subsequent layer.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> with the computed backwards gradient values.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Coth2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arc-cosecant operation with respect to its input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is computed.</param>
    /// <param name="gradient">The gradient of the loss function with respect to the output of the arc-cosecant operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> that stores the gradient with respect to the input tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AcscBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arcsecant (Asec) function for a given tensor during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the values for which the gradient is being calculated.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the incoming gradient in the backpropagation process.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> to store the computed gradient of the Asec function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>, constrained to support IEEE 754 floating point operations and trigonometric functions.</typeparam>
    public void AsecBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arc cotangent function (acot) with respect to its input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for the acot operation.</param>
    /// <param name="gradient">The gradient of the output with respect to a subsequent operation.</param>
    /// <param name="result">The resulting gradient of the acot operation with respect to its input tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AcotBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient for the inverse cosecant squared operation during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the original input values.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the propagated gradient from the next layer.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> for storing the computed gradient of the inverse cosecant squared operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> that supports floating-point and trigonometric operations.</typeparam>
    public void Acsc2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the derivative of the squared arcsecant function with respect to its inputs.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient of the squared arcsecant function is computed.</param>
    /// <param name="gradient">The gradient flowing back from the subsequent operations in the computation graph.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> where the computed gradients are stored.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asec2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the arccotangent function in reverse mode for a given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> to compute the gradient from.</param>
    /// <param name="gradient">The gradient propagated back to this computation.</param>
    /// <param name="result">The resulting <see cref="ITensor{TNumber}"/> containing the computed gradient.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acot2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the inverse hyperbolic cosecant (acsch) operation during backpropagation.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is being computed.</param>
    /// <param name="gradient">The upstream gradient <see cref="ITensor{TNumber}"/> passed from the subsequent layer.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient for the acsch operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AcschBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the inverse hyperbolic secant function (asech) for the given input tensor and gradient tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the asech gradient is to be calculated.</param>
    /// <param name="gradient">The gradient <see cref="ITensor{TNumber}"/> to backpropagate through the asech operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed asech gradient.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>, supporting IEEE 754 floating-point and trigonometric functions.</typeparam>
    public void AsechBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the inverse hyperbolic cotangent function with respect to its input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the values for which the gradient is calculated.</param>
    /// <param name="gradient">The gradient tensor of the subsequent operation.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> storing the calculated gradient of the Acoth function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void AcothBackwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient for the element-wise inverse hyperbolic cosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradient">The gradient tensor representing the backpropagated gradients.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed values.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acsch2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the element-wise inverse hyperbolic secant squared function with respect to the input tensor.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> for which the gradient is being calculated.</param>
    /// <param name="gradient">The gradient tensor representing the upstream gradient contributions.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> containing the computed gradient of the Asech² operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asech2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the gradient of the inverse hyperbolic cotangent operation in the backwards pass for a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/> representing the values for which the gradient is being calculated.</param>
    /// <param name="gradient">The input <see cref="ITensor{TNumber}"/> representing the propagated gradient from the subsequent layer.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/> to store the computed gradient values.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>, supporting floating-point operations and trigonometric functions.</typeparam>
    public void Acoth2Backwards<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> gradient, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;
}