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
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise tangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise sine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise tangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Tan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

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
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosine of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arctangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosine squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arctangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Atan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

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
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise secant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cotangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Csc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise secant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise cotangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Cot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

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
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsecant of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccotangent of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccosecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acsc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arcsecant squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Asec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Computes the element-wise arccotangent squared of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The output <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Acot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>;

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
}