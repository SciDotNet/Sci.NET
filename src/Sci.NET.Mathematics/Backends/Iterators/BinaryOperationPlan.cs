// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Iterators;

internal readonly struct BinaryOperationPlan
{
    private BinaryOperationPlan(
        Shape resultShape,
        long outerIterationCount,
        long innerExtentElements,
        int leftInnerStrideKind,
        int rightInnerStrideKind,
        long leftOuterJumpElements,
        long rightOuterJumpElements)
    {
        ResultShape = resultShape;
        OuterIterationCount = outerIterationCount;
        InnerExtentElements = innerExtentElements;
        LeftInnerStrideKind = leftInnerStrideKind;
        RightInnerStrideKind = rightInnerStrideKind;
        LeftOuterJumpElements = leftOuterJumpElements;
        RightOuterJumpElements = rightOuterJumpElements;
    }

    /// <summary>
    /// Gets the broadcasted result shape (dimensions, strides, element count).
    /// </summary>
    public Shape ResultShape { get; }

    /// <summary>
    /// Gets total number of result elements (shortcut to ResultShape.ElementCount).
    /// </summary>
    public long TotalElementCount => ResultShape.ElementCount;

    /// <summary>
    /// Gets number of outer iterations (collapsed outer product).
    /// </summary>
    public long OuterIterationCount { get; }

    /// <summary>
    /// Gets number of elements in the contiguous inner block per outer iteration.
    /// </summary>
    public long InnerExtentElements { get; }

    /// <summary>
    /// Gets the stride kind for the left inner. 0 = broadcast (no advance), 1 = contiguous advance per element, for the left operand in the inner block.
    /// </summary>
    public int LeftInnerStrideKind { get; }

    /// <summary>
    /// Gets 0 = broadcast, 1 = contiguous, for the right operand in the inner block.
    /// </summary>
    public int RightInnerStrideKind { get; }

    /// <summary>
    /// Gets element offset added to the left operand base after each outer iteration.
    /// </summary>
    public long LeftOuterJumpElements { get; }

    /// <summary>
    /// Gets element offset added to the right operand base after each outer iteration.
    /// </summary>
    public long RightOuterJumpElements { get; }

    /// <summary>
    /// Gets a value indicating whether true if the left operand is fully broadcast across the entire result.
    /// </summary>
    public bool LeftFullyBroadcast => LeftInnerStrideKind == 0 && LeftOuterJumpElements == 0;

    /// <summary>
    /// Gets a value indicating whether true if the right operand is fully broadcast across the entire result.
    /// </summary>
    public bool RightFullyBroadcast => RightInnerStrideKind == 0 && RightOuterJumpElements == 0;

    /// <summary>
    /// Gets a value indicating whether true if the result is a scalar (single element).
    /// </summary>
    public bool IsScalarResult => TotalElementCount == 1;

    public static BinaryOperationPlan Create(Shape leftShape, Shape rightShape)
    {
        // 1. Align & pad dims
        AlignAndPadDimensions(
            leftShape.Dimensions,
            rightShape.Dimensions,
            out int commonRank,
            out int[] leftPadded,
            out int[] rightPadded);

        // 2. Compute result dimensions
        int[] resultDims = ComputeBroadcastShape(leftPadded, rightPadded);

        // 3. Create result shape (canonical contiguous)
        var resultShape = new Shape(resultDims);
        long total = resultShape.ElementCount;

        if (total == 0)
        {
            return new BinaryOperationPlan(
                resultShape,
                outerIterationCount: 0,
                innerExtentElements: 0,
                leftInnerStrideKind: 0,
                rightInnerStrideKind: 0,
                leftOuterJumpElements: 0,
                rightOuterJumpElements: 0);
        }

        // 4. Determine per-axis variation
        var leftVaries = new bool[commonRank];
        var rightVaries = new bool[commonRank];
        for (int axis = 0; axis < commonRank; axis++)
        {
            leftVaries[axis] = leftPadded[axis] > 1;
            rightVaries[axis] = rightPadded[axis] > 1;
        }

        // 5. Select split axis (outer × inner). Current simple heuristic collapses all ⇒ splitAxis = 0.
#pragma warning disable RCS1118 // RCS1118: Use 'var' when type is obvious. We will use this later.
        int splitAxis = 0;
#pragma warning restore RCS1118
        long outerCount = 1;
        long innerExtent = total;

        if (splitAxis > 0)
        {
            outerCount = 1;
            for (int a = 0; a < splitAxis; a++)
            {
                outerCount *= resultDims[a];
            }

            innerExtent = 1;
            for (int a = splitAxis; a < commonRank; a++)
            {
                innerExtent *= resultDims[a];
            }
        }

        // 6. Variation inside inner block
        bool leftVariesInner = false, rightVariesInner = false;
        for (int axis = splitAxis; axis < commonRank; axis++)
        {
            if (leftVaries[axis])
            {
                leftVariesInner = true;
            }

            if (rightVaries[axis])
            {
                rightVariesInner = true;
            }
        }

        var leftInnerStrideKind = leftVariesInner ? 1 : 0;
        var rightInnerStrideKind = rightVariesInner ? 1 : 0;
        var leftOuterJump = 0L;
        var rightOuterJump = 0L;

        if (outerCount > 1)
        {
            long suffix = innerExtent;
            var axisSpanElements = new long[splitAxis];
            for (int axis = splitAxis - 1; axis >= 0; axis--)
            {
                axisSpanElements[axis] = suffix;
                suffix *= resultDims[axis];
            }

            for (int axis = 0; axis < splitAxis; axis++)
            {
                if (leftVaries[axis])
                {
                    leftOuterJump += axisSpanElements[axis];
                }

                if (rightVaries[axis])
                {
                    rightOuterJump += axisSpanElements[axis];
                }
            }
        }

        return new BinaryOperationPlan(
            resultShape,
            outerIterationCount: outerCount,
            innerExtentElements: innerExtent,
            leftInnerStrideKind: leftInnerStrideKind,
            rightInnerStrideKind: rightInnerStrideKind,
            leftOuterJumpElements: leftOuterJump,
            rightOuterJumpElements: rightOuterJump);
    }

    private static void AlignAndPadDimensions(
        ReadOnlySpan<int> leftDims,
        ReadOnlySpan<int> rightDims,
        out int commonRank,
        out int[] leftPadded,
        out int[] rightPadded)
    {
        commonRank = Math.Max(leftDims.Length, rightDims.Length);
        leftPadded = new int[commonRank];
        rightPadded = new int[commonRank];

        int leftOffset = commonRank - leftDims.Length;
        int rightOffset = commonRank - rightDims.Length;

        // Fill leading 1s
        for (int i = 0; i < leftOffset; i++)
        {
            leftPadded[i] = 1;
        }

        for (int i = 0; i < rightOffset; i++)
        {
            rightPadded[i] = 1;
        }

        // Copy originals
        for (int i = 0; i < leftDims.Length; i++)
        {
            leftPadded[leftOffset + i] = leftDims[i];
        }

        for (int i = 0; i < rightDims.Length; i++)
        {
            rightPadded[rightOffset + i] = rightDims[i];
        }
    }

    private static int[] ComputeBroadcastShape(int[] leftPadded, int[] rightPadded)
    {
        int rank = leftPadded.Length;
        var result = new int[rank];
        for (int axis = 0; axis < rank; axis++)
        {
            int l = leftPadded[axis];
            int r = rightPadded[axis];

            result[axis] = l == r
                ? l
                : l switch
                {
                    1 => r,
                    _ => r == 1 ? l : throw new InvalidOperationException($"Shapes are not broadcast-compatible at axis {axis}: {l} vs {r}."),
                };
        }

        return result;
    }
}