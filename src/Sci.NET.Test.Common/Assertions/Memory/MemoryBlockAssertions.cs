// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using FluentAssertions;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using Sci.NET.Common.Memory;

namespace Sci.NET.Test.Common.Assertions.Memory;

/// <summary>
/// Memory block assertions.
/// </summary>
/// <typeparam name="TCollection">The type of the collection.</typeparam>
[DebuggerNonUserCode]
public class MemoryBlockAssertions<TCollection> : ReferenceTypeAssertions<IMemoryBlock<TCollection>,
    MemoryBlockAssertions<TCollection>>
    where TCollection : unmanaged
{
    /// <inheritdoc />
    public MemoryBlockAssertions(IMemoryBlock<TCollection> subject)
        : base(subject)
    {
    }

    /// <inheritdoc />
    protected override string Identifier => nameof(MemoryBlockAssertions<TCollection>);

    /// <summary>
    /// Asserts that the memory block is equal to the expected values.
    /// </summary>
    /// <param name="expectedValues">The expected values.</param>
    /// <param name="because">The reason for the expectation.</param>
    /// <param name="becauseArgs">The the formatting values for the <paramref name="because"/> string.</param>
    /// <returns>An <see cref="AndConstraint{T}"/> for chained assertions.</returns>
    public AndConstraint<MemoryBlockAssertions<TCollection>> BeEqualTo(
        TCollection[] expectedValues,
        string because = "",
        params object[] becauseArgs)
    {
        _ = Execute.Assertion
            .BecauseOf(because, becauseArgs)
            .Given(() => Subject.Length)
            .ForCondition(length => length == expectedValues.Length)
            .FailWith("Expected Result to have length {0} but has {1}", expectedValues.Length, Subject.Length);

        _ = Execute.Assertion
            .BecauseOf(because, becauseArgs)
            .Given(() => Subject)
            .ForCondition(
                status =>
                {
                    for (var i = 0; i < status.Length; i++)
                    {
                        if (!status[i].Equals(expectedValues[i]))
                        {
                            return false;
                        }
                    }

                    return true;
                })
            .FailWith("Expected Result to have values {0} but has {1}", expectedValues.ToArray(), Subject.ToArray());

        return new AndConstraint<MemoryBlockAssertions<TCollection>>(this);
    }
}