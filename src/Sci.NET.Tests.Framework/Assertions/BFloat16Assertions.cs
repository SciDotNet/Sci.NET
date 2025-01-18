// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using FluentAssertions.Execution;
using FluentAssertions.Numeric;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Tests.Framework.Assertions;

/// <summary>
/// Assertions for <see cref="BFloat16" />.
/// </summary>
public class BFloat16Assertions : NumericAssertions<BFloat16, BFloat16Assertions>
{
    private readonly AssertionChain _assertionChain;

    /// <summary>
    /// Initializes a new instance of the <see cref="BFloat16Assertions"/> class.
    /// </summary>
    /// <param name="value">The value to create assertions for.</param>
    /// <param name="chain">The chain of assertions.</param>
    public BFloat16Assertions(BFloat16 value, AssertionChain chain)
        : base(value, chain)
    {
        _assertionChain = chain;
    }

    /// <summary>
    /// Asserts that the value is approximately equal to the expected value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="because">The reason why the assertion is needed. If the phrase does not start with the word <i>because</i>, it is prepended automatically.</param>
    /// <param name="becauseArgs">Zero or more objects to format using the placeholders in <paramref name="because" />.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    [PublicAPI]
    [CustomAssertion]
    public new AndConstraint<BFloat16Assertions> Be(BFloat16 expected, string because = "", params object[] becauseArgs)
    {
        _ = _assertionChain
            .ForCondition(Subject.Equals(expected))
            .BecauseOf(because, becauseArgs)
            .FailWith("Expected {context:value} to be approximately {0}{reason}, but found {1}.", expected, Subject);

        return new AndConstraint<BFloat16Assertions>(this);
    }

    /// <summary>
    /// Asserts that the value is approximately equal to the expected value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="tolerance">The tolerance within which the value is expected to be.</param>
    /// <param name="because">The reason why the assertion is needed. If the phrase does not start with the word <i>because</i>, it is prepended automatically.</param>
    /// <param name="becauseArgs">Zero or more objects to format using the placeholders in <paramref name="because" />.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the subject is <see langword="null" /> or not comparable (I.E. infinite, NaN).</exception>
    [PublicAPI]
    [CustomAssertion]
    public AndConstraint<BFloat16Assertions> BeApproximately(BFloat16 expected, float tolerance, string because = "", params object[] becauseArgs)
    {
        if (BFloat16.IsNaN(Subject))
        {
            throw new InvalidOperationException("Cannot assert that a NaN value is approximately equal to another value.");
        }

        if (BFloat16.IsNaN(expected))
        {
            throw new InvalidOperationException("Cannot assert that a value is approximately equal to a NaN value.");
        }

        if (BFloat16.IsPositiveInfinity(Subject) || BFloat16.IsNegativeInfinity(Subject))
        {
            throw new InvalidOperationException("Cannot assert that an infinity value is approximately equal to another value.");
        }

        if (BFloat16.IsPositiveInfinity(expected) || BFloat16.IsNegativeInfinity(expected))
        {
            throw new InvalidOperationException("Cannot assert that a value is approximately equal to an infinity value.");
        }

        _ = _assertionChain
            .ForCondition(float.Abs((float)Subject - (float)expected) <= tolerance)
            .BecauseOf(because, becauseArgs)
            .FailWith(
                "Expected {context:value} to be approximately {0} +/- {1}{reason}, but found {2}.",
                expected,
                tolerance,
                Subject);

        return new AndConstraint<BFloat16Assertions>(this);
    }
}