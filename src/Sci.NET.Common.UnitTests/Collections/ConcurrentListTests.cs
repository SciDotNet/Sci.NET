// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Collections;
using Sci.NET.Common.Concurrency;

namespace Sci.NET.Common.UnitTests.Collections;

public class ConcurrentListTests
{
    [Fact]
    public void Add_AddsItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        list.ElementAt(0).Should().Be(1);
        list.Count.Should().Be(1);

        rwl.Verify(r => r.AcquireWriterLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseWriterLock(), Times.Once);
    }

    [Fact]
    public void AddRange_AddsItems()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.AddRange(new int[] { 1, 2, 3 });

        list.ElementAt(0).Should().Be(1);
        list.ElementAt(1).Should().Be(2);
        list.ElementAt(2).Should().Be(3);
        list.Count.Should().Be(3);

        rwl.Verify(r => r.AcquireWriterLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseWriterLock(), Times.Once);
    }

    [Fact]
    public void Contains_ReturnsTrueForContainedItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        list.Contains(1).Should().BeTrue();

        rwl.Verify(r => r.AcquireReaderLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseReaderLock(), Times.Once);
    }

    [Fact]
    public void Contains_ReturnsFalseForNonContainedItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        list.Contains(2).Should().BeFalse();

        rwl.Verify(r => r.AcquireReaderLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseReaderLock(), Times.Once);
    }

    [Fact]
    public void ReplaceAt_ReplacesItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        rwl.Invocations.Clear();

        list.ReplaceAt(0, 2);

        rwl.Verify(r => r.AcquireWriterLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseWriterLock(), Times.Once);

        list.ElementAt(0).Should().Be(2);
    }

    [Fact]
    public void ElementAt_ReturnsCorrectItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        list.ElementAt(0).Should().Be(1);

        rwl.Verify(r => r.AcquireReaderLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseReaderLock(), Times.Once);
    }

    [Fact]
    public void IndexOf_ReturnsCorrectIndex()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        rwl.Invocations.Clear();

        list.Remove(1).Should().BeTrue();

        rwl.Verify(r => r.AcquireWriterLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseWriterLock(), Times.Once);

        list.Count.Should().Be(0);
    }

    [Fact]
    public void Insert_InsertsItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        rwl.Invocations.Clear();

        list.Insert(0, 2);

        rwl.Verify(r => r.AcquireWriterLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseWriterLock(), Times.Once);

        list.ElementAt(0).Should().Be(2);
    }

    [Fact]
    public void Indexer_SetItem_SetsItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        rwl.Invocations.Clear();

        list[0] = 2;

        rwl.Verify(r => r.AcquireWriterLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseWriterLock(), Times.Once);

        list.ElementAt(0).Should().Be(2);
    }

    [Fact]
    public void Indexer_GetItem_GetsItem()
    {
        var rwl = new Mock<IReaderWriterLock>();
        var list = new ConcurrentList<int>(rwl.Object);

        list.Add(1);

        rwl.Invocations.Clear();

        list[0].Should().Be(1);

        rwl.Verify(r => r.AcquireReaderLock(It.Is<TimeSpan>(x => x == Timeout.InfiniteTimeSpan)), Times.Once);
        rwl.Verify(r => r.ReleaseReaderLock(), Times.Once);
    }
}