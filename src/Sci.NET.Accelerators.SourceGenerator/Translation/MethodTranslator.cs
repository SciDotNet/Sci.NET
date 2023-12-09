// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerator.Diagnostics;
using Sci.NET.Accelerators.SourceGenerator.Translation.Visitors;

namespace Sci.NET.Accelerators.SourceGenerator.Translation;

internal static class MethodTranslator
{
    public static void Translate(TranslationContext translationContext)
    {
        translationContext.ParameterListSyntax = ValidateAndModifyParameters(translationContext);
        translationContext.BlockSyntax = ModifyMethodBlock(translationContext);
    }

    private static BlockSyntax ModifyMethodBlock(TranslationContext translationContext)
    {
        if (translationContext.MethodDeclarationSyntax.Body is null)
        {
            TranslationDiagnostics.ReportMissingBody(translationContext.SourceProductionContext, translationContext.MethodDeclarationSyntax);
            return SyntaxFactory.Block();
        }

        var blockVisitor = new BlockSyntaxVisitor();
        return blockVisitor.Visit(translationContext.MethodDeclarationSyntax.Body, translationContext);
    }

    private static ParameterListSyntax ValidateAndModifyParameters(TranslationContext translationContext)
    {
        var parameters = new List<ParameterSyntax>();

        foreach (var parameter in translationContext.MethodDeclarationSyntax.ParameterList.Parameters)
        {
            if (parameter.Modifiers.Any())
            {
                TranslationDiagnostics.ReportUnsupportedParameterModifier(translationContext.SourceProductionContext, parameter);
            }

            if (parameter.Type?.IsUnmanaged ?? false)
            {
                parameters.Add(SyntaxFactory.Parameter(parameter.Identifier).WithType(parameter.Type));
                continue;
            }

            if (parameter.Type is not GenericNameSyntax genericNameSyntax)
            {
                TranslationDiagnostics.ReportUnsupportedParameterType(translationContext.SourceProductionContext, parameter);
                continue;
            }

            if (genericNameSyntax.Identifier.Text != "IMemoryBlock")
            {
                TranslationDiagnostics.ReportUnsupportedParameterType(translationContext.SourceProductionContext, parameter);
                continue;
            }

            if (genericNameSyntax.TypeArgumentList.Arguments.Count != 1)
            {
                TranslationDiagnostics.ReportUnsupportedParameterType(translationContext.SourceProductionContext, parameter);
                continue;
            }

            parameters.Add(
                SyntaxFactory
                    .Parameter(parameter.Identifier)
                    .WithType(SyntaxFactory.PointerType(genericNameSyntax.TypeArgumentList.Arguments[0])));

            parameters.Add(
                SyntaxFactory
                    .Parameter(SyntaxFactory.Identifier($"{parameter.Identifier.Text}Length"))
                    .WithType(SyntaxFactory.PredefinedType(SyntaxFactory.Token(SyntaxKind.LongKeyword))));
        }

        return SyntaxFactory.ParameterList(SyntaxFactory.SeparatedList(parameters));
    }
}