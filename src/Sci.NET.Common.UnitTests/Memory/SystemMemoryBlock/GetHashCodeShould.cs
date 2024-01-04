// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class GetHashCodeShould
{
    [Fact]
    public void BeUnique_ForDifferentInstances()
    {
        var block1 = new SystemMemoryBlock<int>(10);
        var block2 = new SystemMemoryBlock<int>(10);

        block1.GetHashCode().Should().NotBe(block2.GetHashCode());
    }
}