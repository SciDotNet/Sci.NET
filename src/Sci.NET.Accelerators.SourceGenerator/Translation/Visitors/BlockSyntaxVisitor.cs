// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Statements;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors;

internal class BlockSyntaxVisitor : ITranslationVisitor<BlockSyntax>
{
    public BlockSyntax Visit(BlockSyntax syntaxNode, TranslationContext context)
    {
        var statements = syntaxNode.Statements.Select(statement => new StatementVisitor().Visit(statement, context)).ToList();

        return SyntaxFactory.Block(statements);
    }
}