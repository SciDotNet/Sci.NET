// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class GetReferenceShould
{
    [Fact]
    public void ReturnCorrectReference_GivenValidInstance()
    {
        var memoryBlock = new SystemMemoryBlock<int>(10);

        var reference = memoryBlock.GetReference();

        reference.Should().Be(memoryBlock[0]);
    }

    [Fact]
    public void ThrowException_WhenDisposed()
    {
        var memoryBlock = new SystemMemoryBlock<int>(10);
        memoryBlock.Dispose();

        var act = () => memoryBlock.GetReference();

        act.Should().Throw<ObjectDisposedException>();
    }
}