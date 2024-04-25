// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

/// <summary>
/// A manager for instruction transforms.
/// </summary>
[PublicAPI]
public static class TransformManager
{
    private static readonly ConcurrentQueue<IIrTransform> _builtinTransforms;
    private static readonly ConcurrentQueue<IIrTransform> _userTransforms;

#pragma warning disable CA1810
    static TransformManager()
#pragma warning restore CA1810
    {
        _builtinTransforms = new ConcurrentQueue<IIrTransform>();
        _userTransforms = new ConcurrentQueue<IIrTransform>();

        // Add built-in transforms
        AddBuiltinTransform(new ThreadIndexTransform());
        AddBuiltinTransform(new RemoveUnusedInstructions());
    }

    /// <summary>
    /// Adds a transform to the manager.
    /// </summary>
    /// <param name="transform">The transform to add.</param>
    public static void AddTransform(IIrTransform transform)
    {
        _userTransforms.Enqueue(transform);
    }

    /// <summary>
    /// Clears all custom transforms.
    /// </summary>
    public static void ClearCustomTransforms()
    {
        _userTransforms.Clear();
    }

    /// <summary>
    /// Gets all transforms.
    /// </summary>
    /// <returns>All transforms.</returns>
    public static IEnumerable<IIrTransform> GetTransforms()
    {
        return _builtinTransforms.Concat(_userTransforms);
    }

    internal static void AddBuiltinTransform(IIrTransform transform)
    {
        _builtinTransforms.Enqueue(transform);
    }

    internal static void ClearBuiltinTransforms()
    {
        _builtinTransforms.Clear();
    }
}