// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using FluentAssertions;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Test.Common.Assertions.Tensors;

/// <summary>
/// Contains a number of methods to assert that an <see cref="ITensor{TNumber}"/> is in the expected state.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
[ExcludeFromCodeCoverage]
[DebuggerNonUserCode]
public class TensorAssertions<TNumber> : ReferenceTypeAssertions<ITensor<TNumber>,
    TensorAssertions<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <inheritdoc />
    public TensorAssertions(ITensor<TNumber> subject)
        : base(subject)
    {
    }

    /// <inheritdoc />
    protected override string Identifier => nameof(TensorAssertions<TNumber>);

    /// <summary>
    /// Asserts that the <see cref="ITensor{TNumber}"/> is equal to the other <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="other">The expected value.</param>
    /// <param name="because">The reason for the expectation.</param>
    /// <param name="becauseArgs">The formatting values for the <paramref name="because"/> string.</param>
    /// <returns>An <see cref="AndConstraint{T}"/> for chained assertions.</returns>
    public AndConstraint<TensorAssertions<TNumber>> BeEqualTo(
        ITensor<TNumber> other,
        string because = "",
        params object[] becauseArgs)
    {
        _ = Execute.Assertion
            .BecauseOf(because, becauseArgs)
            .Given(() => Subject.GetShape())
            .ForCondition(shape => shape == other.GetShape())
            .FailWith("Expected Result to have length {0} but has {1}", other.GetShape(), Subject.GetShape());

        _ = Execute.Assertion
            .BecauseOf(because, becauseArgs)
            .Given(() => Subject)
            .ForCondition(
                status =>
                {
                    for (var i = 0; i < status.Data.Length; i++)
                    {
                        if (!status.Data[i].Equals(other.Data[i]))
                        {
                            return false;
                        }
                    }

                    return true;
                })
            .FailWith("Expected Result to be equal to {0} but found {1}", other.Data, Subject.Data);

        return new AndConstraint<TensorAssertions<TNumber>>(this);
    }
}