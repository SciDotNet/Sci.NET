// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.DeepLearning.VariableInitialization;

/// <summary>
/// A class for variable initializers.
/// </summary>
[PublicAPI]
public static class FloatingPointVariableInitializer
{
    /// <summary>
    /// Gets a default <see cref="VariableInitializer{TNumber}"/> for the specified number type.
    /// </summary>
    /// <typeparam name="TNumber">The number type to create.</typeparam>
    /// <returns>The default variable initializer.</returns>
    public static VariableInitializer<TNumber> GetDefault<TNumber>()
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        return new VariableInitializer<TNumber>(
            TNumber.CreateChecked(0.0001),
            TNumber.CreateChecked(0.01),
            DateTime.UtcNow.Ticks);
    }
}