// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Expression;

internal class MemberAccessExpressionSyntaxVisitor : ITranslationVisitor<MemberAccessExpressionSyntax>
{
    public MemberAccessExpressionSyntax Visit(MemberAccessExpressionSyntax syntaxNode, TranslationContext context)
    {
        if (syntaxNode.Expression is not IdentifierNameSyntax identifierNameSyntax)
        {
            return syntaxNode;
        }

        var semanticModel = context.Compilation.GetSemanticModel(identifierNameSyntax.SyntaxTree);
        var operation = semanticModel.GetOperation(syntaxNode);

        if (context.Compilation.GetSemanticModel(identifierNameSyntax.SyntaxTree).GetOperation(syntaxNode) is not IPropertyReferenceOperation propertyReferenceOperation)
        {
            return syntaxNode;
        }

        if (propertyReferenceOperation.Property.Name != "Length")
        {
            return syntaxNode;
        }

        if (context.ParameterListSyntax.Parameters.Any(x => x.Identifier.Text == identifierNameSyntax.Identifier.Text))
        {
            return syntaxNode.WithExpression(SyntaxFactory.IdentifierName(identifierNameSyntax.Identifier.Text + "Length"));
        }

        return syntaxNode;
    }
}