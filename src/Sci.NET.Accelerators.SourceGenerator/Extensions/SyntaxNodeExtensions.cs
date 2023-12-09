// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;

namespace Sci.NET.Accelerators.SourceGenerator.Extensions;

internal static class SyntaxNodeExtensions
{
    public static IEnumerable<SyntaxNode> RecursiveChildren(this SyntaxNode syntaxNode)
    {
        foreach (var child in syntaxNode.ChildNodes())
        {
            yield return child;

            foreach (var recursiveChild in child.RecursiveChildren())
            {
                yield return recursiveChild;
            }
        }
    }
}