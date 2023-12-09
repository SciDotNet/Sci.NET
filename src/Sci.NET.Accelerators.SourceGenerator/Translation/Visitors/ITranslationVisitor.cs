// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors;

/// <summary>
/// A visitor that modifies a syntax node.
/// </summary>
/// <typeparam name="TSyntaxNode">The type of syntax node to visit.</typeparam>
internal interface ITranslationVisitor<TSyntaxNode>
    where TSyntaxNode : SyntaxNode
{
    /// <summary>
    /// Visits a syntax node.
    /// </summary>
    /// <param name="syntaxNode">The syntax node to visit.</param>
    /// <param name="context">The translation context.</param>
    /// <returns>The modified syntax node.</returns>
    public TSyntaxNode Visit(TSyntaxNode syntaxNode, TranslationContext context);
}