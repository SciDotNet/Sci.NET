// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Sci.NET.Common.Runtime;

#pragma warning disable CA1060

/// <summary>
/// A helper class for resolving native library paths.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
public static class RuntimeDllImportResolver
{
    /// <summary>
    /// Loads a native library.
    /// </summary>
    /// <param name="libraryName">The library name to look for.</param>
    /// <param name="assembly">The assembly to load to.</param>
    /// <returns>A pointer to the loaded library.</returns>
    /// <exception cref="InvalidOperationException">The library failed to load.</exception>
    /// <exception cref="PlatformNotSupportedException">The current platform is not supported.</exception>
    public static nint LoadLibrary(string libraryName, Assembly assembly)
    {
        var assemblyDirectory = Path.GetDirectoryName(assembly.Location) ??
                                throw new InvalidOperationException("The assembly directory could not be resolved.");

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (TryLoadWindowsLibrary(libraryName, assemblyDirectory, assembly, out var handle))
            {
                return handle;
            }

            throw new InvalidOperationException("The native library could not be loaded.");
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (TryLoadLinuxLibrary(libraryName, assemblyDirectory, assembly, out var handle))
            {
                return handle;
            }

            throw new InvalidOperationException("The native library could not be loaded.");
        }

        throw new PlatformNotSupportedException("This platform is not supported.");
    }

    private static bool TryLoadLinuxLibrary(string libraryName, string assemblyDirectory, Assembly assembly, out nint o)
    {
        return NativeLibrary.TryLoad($@"{assemblyDirectory}\runtimes\linux-x64\{libraryName}.so", assembly, null, out o) ||
               NativeLibrary.TryLoad($@"{assemblyDirectory}\{libraryName}.so", assembly, null, out o) ||
               NativeLibrary.TryLoad($@"{assemblyDirectory}\{libraryName}.so.1", assembly, null, out o);
    }

    private static bool TryLoadWindowsLibrary(string libraryName, string assemblyDirectory, Assembly assembly, out nint o)
    {
        return NativeLibrary.TryLoad($@"{assemblyDirectory}\CUDA\win\x64\{libraryName}.dll", assembly, null, out o) ||
               NativeLibrary.TryLoad($@"{assemblyDirectory}\CUDA\win-x64\{libraryName}.dll", assembly, null, out o) ||
               NativeLibrary.TryLoad($@"{assemblyDirectory}\{libraryName}.dll", assembly, null, out o);
    }
}
#pragma warning restore CA1060