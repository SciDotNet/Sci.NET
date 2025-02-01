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
    /// Calculates the sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The tangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the sine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the sine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sin2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the cosine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The cosine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Cos2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the tangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the tangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The tangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Tan2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic sine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sinh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Cosh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic tangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Tanh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic sine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic sine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic sine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sinh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic cosine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic cosine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Cosh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic tangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic tangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic tangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Tanh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse sine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Asin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse cosine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Acos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse tangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Atan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic sine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ASinh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic cosine of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACosh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic tangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ATanh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse sine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse sine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse sine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Asin2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse cosine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse cosine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse cosine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Acos2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse tangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse tangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse tangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Atan2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic sine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic sine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic sine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ASinh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic cosine squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic cosine squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic cosine squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACosh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic tangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic tangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic tangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ATanh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The cosecant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Csc<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The secant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sec<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The cotangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Cot<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cosecant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the cosecant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The cosecant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Csc2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the secant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the secant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The secant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sec2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the cotangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the cotangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The cotangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Cot2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cotangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Csch<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic secant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sech<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cotangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Coth<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic cosecant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic cosecant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cosecant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Csch2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic secant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic secant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic secant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sech2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the hyperbolic cotangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the hyperbolic cotangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The hyperbolic cotangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Coth2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse cosecant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Acsc<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse secant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Asec<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse cotangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Acot<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse cosecant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse cosecant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse cosecant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACsc2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse secant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse secant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse secant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ASec2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse cotangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse cotangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse cotangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACot2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic cosecant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACsch<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic secant of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ASech<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic cotangent of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACoth<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic cosecant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic cosecant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic cosecant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACsch2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic secant squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic secant squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic secant squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ASech2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Calculates the inverse hyperbolic cotangent squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to calculate the inverse hyperbolic cotangent squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The inverse hyperbolic cotangent squared of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ACoth2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>;
}