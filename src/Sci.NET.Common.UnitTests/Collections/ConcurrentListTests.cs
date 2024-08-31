// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Collections;

namespace Sci.NET.Common.UnitTests.Collections;

public class ConcurrentListTests
{
    [Fact]
    public void Add_AddsItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list.ElementAt(0).Should().Be(1);
        list.Count.Should().Be(1);
    }

    [Fact]
    public void AddRange_AddsItems()
    {
        var list = new ConcurrentList<int>();

        list.AddRange(new int[] { 1, 2, 3 });

        list.ElementAt(0).Should().Be(1);
        list.ElementAt(1).Should().Be(2);
        list.ElementAt(2).Should().Be(3);
        list.Count.Should().Be(3);
    }

    [Fact]
    public void Contains_ReturnsTrueForContainedItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list.Contains(1).Should().BeTrue();
    }

    [Fact]
    public void Contains_ReturnsFalseForNonContainedItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list.Contains(2).Should().BeFalse();
    }

    [Fact]
    public void ReplaceAt_ReplacesItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);
        list.ReplaceAt(0, 2);
        list.ElementAt(0).Should().Be(2);
    }

    [Fact]
    public void ElementAt_ReturnsCorrectItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list.ElementAt(0).Should().Be(1);
    }

    [Fact]
    public void IndexOf_ReturnsCorrectIndex()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list.Remove(1).Should().BeTrue();

        list.Count.Should().Be(0);
    }

    [Fact]
    public void Insert_InsertsItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list.Insert(0, 2);

        list.ElementAt(0).Should().Be(2);
    }

    [Fact]
    public void Indexer_SetItem_SetsItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list[0] = 2;

        list.ElementAt(0).Should().Be(2);
    }

    [Fact]
    public void Indexer_GetItem_GetsItem()
    {
        var list = new ConcurrentList<int>();

        list.Add(1);

        list[0].Should().Be(1);
    }
}