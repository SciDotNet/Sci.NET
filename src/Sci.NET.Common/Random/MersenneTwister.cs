// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.LowLevel;

namespace Sci.NET.Common.Random;

/// <summary>
/// A Mersenne Twister random number generator.
/// </summary>
[PublicAPI]
public sealed class MersenneTwister
{
    private const int N = 624;
    private const int M = 397;
    private const uint MatrixA = 0x9908b0df;
    private const uint UpperMask = 0x80000000;
    private const uint LowerMask = 0x7fffffff;

    private static readonly uint[] Mag01 = { 0x0, MatrixA };

    private static readonly ThreadLocal<MersenneTwister> ThreadLocalInstance =
        new (() => new MersenneTwister(GenerateUniqueSeed()));

    private static readonly Lazy<MersenneTwister> ThisInstance = new (() => ThreadLocalInstance.Value!);

    private uint[] _mt;
    private int _index;

    private MersenneTwister(uint seed)
    {
        _mt = new uint[N];
        Seed(seed);
    }

    /// <summary>
    /// Gets the global singleton instance of the Mersenne Twister random number generator.
    /// </summary>
    public static MersenneTwister Instance
    {
        get
        {
            lock (ThisInstance)
            {
                return ThisInstance.Value;
            }
        }
    }

    /// <summary>
    /// Generates a random number between 0 and 2^32-1.
    /// </summary>
    /// <returns>A random number between 0 and 2^32-1.</returns>
    public uint NextUInt()
    {
        if (_index >= N)
        {
            GenerateNumbers();
        }

        var y = _mt[_index++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;

        return y;
    }

    /// <summary>
    /// Generates a random <see cref="int"/>.
    /// </summary>
    /// <returns>A random <see cref="int"/>.</returns>
    public int NextInt()
    {
        return (int)(NextUInt() >> 1);
    }

    /// <summary>
    /// Generates a random <see cref="int"/> less than <paramref name="maxValue"/>.
    /// </summary>
    /// <param name="maxValue">The maximum value to generate.</param>
    /// <returns>A random number less than <paramref name="maxValue"/>.</returns>
    public uint NextUInt(uint maxValue)
    {
        return NextUInt() % maxValue;
    }

    /// <summary>
    /// Generates a random <see cref="int"/> between <paramref name="minValue"/> and <paramref name="maxValue"/>.
    /// </summary>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <returns>A random number between the two constraints.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The maximum is smaller than the minimum.</exception>
    public int NextInt(int minValue, int maxValue)
    {
        if (minValue >= maxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(minValue), "minValue must be less than maxValue.");
        }

        var next = NextUInt(int.Abs(maxValue - minValue).ReinterpretCast<int, uint>());

        return minValue + next.ReinterpretCast<uint, int>();
    }

    /// <summary>
    /// Generates a random <see cref="double"/> between zero and one.
    /// </summary>
    /// <returns>A random double between zero and one.</returns>
    public double NextDouble()
    {
        const double divisor = 1.0 / (1UL << 32);
        return NextUInt() * divisor;
    }

    /// <summary>
    /// Generates a random <see cref="float"/> between <paramref name="minValue"/> and <paramref name="maxValue"/>.
    /// </summary>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <returns>A random number between the two constraints.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The maximum is smaller than the minimum.</exception>
    public float NextFloat(float minValue, float maxValue)
    {
        if (minValue >= maxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(minValue), "minValue must be less than maxValue.");
        }

        var range = (double)maxValue - minValue;
        var randomValue = (double)NextUInt() / uint.MaxValue;

        return (float)(minValue + (randomValue * range));
    }

    /// <summary>
    /// Generates a random <see cref="double"/> between <paramref name="minValue"/> and <paramref name="maxValue"/>.
    /// </summary>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <returns>A random number between the two constraints.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The maximum is smaller than the minimum.</exception>
    public double NextDouble(double minValue, double maxValue)
    {
        if (minValue >= maxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(minValue), "minValue must be less than maxValue.");
        }

        var range = maxValue - minValue;
        var randomValue = (double)NextUInt() / uint.MaxValue;

        return minValue + (randomValue * range);
    }

    /// <summary>
    /// Seeds the Random Number Generator.
    /// </summary>
    /// <param name="seed">The seed.</param>
    public void Seed(uint seed)
    {
        _mt = new uint[N];
        _mt[0] = seed;

        for (var i = 1; i < N; i++)
        {
            _mt[i] = (1812433253 * (_mt[i - 1] ^ (_mt[i - 1] >> 30))) + (uint)i;
        }
    }

    private static uint GenerateUniqueSeed()
    {
        lock (ThisInstance)
        {
            // Generate a random seed using the global singleton instance
            return ThisInstance.Value.NextUInt();
        }
    }

    private void GenerateNumbers()
    {
        for (var i = 0; i < N; i++)
        {
            var y = (_mt[i] & UpperMask) | (_mt[(i + 1) % N] & LowerMask);
            _mt[i] = _mt[(i + M) % N] ^ (y >> 1) ^ Mag01[y & 0x1];
        }

        _index = 0;
    }
}