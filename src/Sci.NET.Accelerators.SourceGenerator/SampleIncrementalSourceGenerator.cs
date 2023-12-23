// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerator.Translation;

namespace Sci.NET.Accelerators.SourceGenerator;

/// <summary>
/// A sample source generator that creates a custom report based on class properties. The target class should be annotated with the 'Generators.ReportAttribute' attribute.
/// When using the source code as a baseline, an incremental source generator is preferable because it reduces the performance overhead.
/// </summary>
[Generator]
public class SampleIncrementalSourceGenerator : IIncrementalGenerator
{
    private const string Namespace = "Sci.NET.Accelerators.Attributes";
    private const string AttributeName = "KernelAttribute";

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var provider = context
            .SyntaxProvider
            .CreateSyntaxProvider(
                (s, _) => s is MethodDeclarationSyntax,
                (ctx, _) => GetMethodDeclarationForSourceGen(ctx))
            .Where(t => t.KernelAttributeFound)
            .Select((t, _) => t.Method);

        context.RegisterSourceOutput(
            context.CompilationProvider.Combine(provider.Collect()),
            (ctx, t) => TranslateAndGenerateCode(ctx, t.Left, t.Right));
    }

    /// <summary>
    /// Checks whether the node is a <see cref="MethodDeclarationSyntax"/> and is annotated with the
    /// [Kernel] attribute and maps syntax context to the specific node type (MethodDeclarationSyntax).
    /// </summary>
    /// <param name="context">Syntax context, based on CreateSyntaxProvider predicate.</param>
    /// <returns>The specific cast and whether the attribute was found.</returns>
    private static (MethodDeclarationSyntax Method, bool KernelAttributeFound) GetMethodDeclarationForSourceGen(GeneratorSyntaxContext context)
    {
        var methodDeclarationSyntax = (MethodDeclarationSyntax)context.Node;

        foreach (var methodSyntax in methodDeclarationSyntax.AttributeLists.SelectMany(attributeListSyntax => attributeListSyntax.Attributes))
        {
            if (ModelExtensions.GetSymbolInfo(context.SemanticModel, methodSyntax).Symbol is not IMethodSymbol attributeSymbol)
            {
                continue;
            }

            var attributeName = attributeSymbol.ContainingType.ToDisplayString();

            if (attributeName == $"{Namespace}.{AttributeName}")
            {
                return (methodDeclarationSyntax, true);
            }
        }

        return (methodDeclarationSyntax, false);
    }

    /// <summary>
    /// Generate code action.
    /// It will be executed on specific nodes (MethodDeclarationSyntax annotated with the [Kernel] attribute) changed by the user.
    /// </summary>
    /// <param name="context">Source generation context used to add source files.</param>
    /// <param name="compilation">Compilation used to provide access to the Semantic Model.</param>
    /// <param name="methodDeclarations">Nodes annotated with the [Kernel] attribute that trigger the generate action.</param>
    private static void TranslateAndGenerateCode(
        SourceProductionContext context,
        Compilation compilation,
        ImmutableArray<MethodDeclarationSyntax> methodDeclarations)
    {
        foreach (var methodDeclarationSyntax in methodDeclarations)
        {
            var translationContext = new TranslationContext(methodDeclarationSyntax, context, compilation);
            MethodTranslator.Translate(translationContext);

            _ = SyntaxFactory
                .MethodDeclaration(
                    SyntaxFactory.PredefinedType(SyntaxFactory.Token(SyntaxKind.VoidKeyword)),
                    SyntaxFactory.Identifier("Invoke" + methodDeclarationSyntax.Identifier.Text))
                .WithModifiers(SyntaxFactory.TokenList(SyntaxFactory.Token(SyntaxKind.PublicKeyword)))
                .WithParameterList(translationContext.ParameterListSyntax)
                .WithBody(translationContext.EntryBlockSyntax);

            // context.AddSource(methodDeclarationSyntax.Identifier.Text, methodDeclaration.NormalizeWhitespace().ToFullString());
        }
    }
}