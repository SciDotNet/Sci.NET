// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Common.LowLevel;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Mathematics.Random;

/// <summary>
/// A deterministic random number generator.
/// </summary>
[PublicAPI]
[SuppressMessage("Security", "CA5394:Do not use insecure randomness", Justification = "This is not for security purposes.")]
public sealed class DeterministicRandomNumberGenerator : IDisposable
{
    private readonly System.Random _globalRandom;
    private readonly ThreadLocal<System.Random> _localRandom;

    /// <summary>
    /// Initializes a new instance of the <see cref="DeterministicRandomNumberGenerator"/> class.
    /// </summary>
    /// <param name="seed">The seed for the random number generator.</param>
    public DeterministicRandomNumberGenerator(int? seed = null)
    {
        _globalRandom = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
        _localRandom = new ThreadLocal<System.Random>(() => new System.Random(_globalRandom.Next()));
    }

    /// <summary>
    /// Generates a random number.
    /// </summary>
    /// <param name="min">The minimum value of the distribution.</param>
    /// <param name="max">The maximum value of the distribution.</param>
    /// <typeparam name="TNumber">The type of the number to generate.</typeparam>
    /// <returns>A random number.</returns>
    /// <exception cref="NotSupportedException">Thrown if the number type is not supported.</exception>
    public TNumber Next<TNumber>(TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var rng = _localRandom.Value!;

        return TNumber.Zero switch
        {
            BFloat16 => min + ((max - min) * TNumber.CreateChecked((BFloat16)rng.NextSingle())),
            float => min + ((max - min) * TNumber.CreateChecked(rng.NextSingle())),
            double => min + ((max - min) * TNumber.CreateChecked(rng.NextDouble())),
            byte => min + ((max - min) * TNumber.CreateChecked((byte)rng.Next(byte.MinValue, byte.MaxValue))),
            ushort => min + ((max - min) * TNumber.CreateChecked((ushort)rng.Next(ushort.MinValue, ushort.MaxValue))),
            uint => min + ((max - min) * TNumber.CreateChecked(rng.Next(uint.MinValue.ReinterpretCast<uint, int>(), int.MaxValue).ReinterpretCast<int, uint>())),
            ulong => min + ((max - min) * TNumber.CreateChecked(rng.NextInt64(ulong.MinValue.ReinterpretCast<ulong, long>(), ulong.MaxValue.ReinterpretCast<ulong, long>()).ReinterpretCast<long, ulong>())),
            sbyte => min + ((max - min) * TNumber.CreateChecked((sbyte)rng.Next())),
            short => min + ((max - min) * TNumber.CreateChecked((short)rng.Next())),
            int => min + ((max - min) * TNumber.CreateChecked(rng.Next())),
            long => min + ((max - min) * TNumber.CreateChecked(rng.NextInt64())),
            _ => throw new NotSupportedException("Unsupported number type.")
        };
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _localRandom.Dispose();
    }
}