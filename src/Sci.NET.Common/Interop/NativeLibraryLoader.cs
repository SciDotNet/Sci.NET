// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Sci.NET.Common.Interop;

/// <summary>
/// Provides methods to load native libraries.
/// </summary>
public static class NativeLibraryLoader
{
    private static readonly ConcurrentDictionary<string, nint> NativeHandles = new();

    /// <summary>
    /// Try to load a native library.
    /// </summary>
    /// <param name="libraryName">The name of the native library.</param>
    /// <param name="assembly">The assembly to load into.</param>
    /// <param name="searchDirectory">The search directory.</param>
    /// <returns>A value indicating whether loading was successful.</returns>
    public static bool TryLoad(string libraryName, Assembly assembly, DllImportSearchPath searchDirectory)
    {
#pragma warning disable IDE0046
        var os = GetOsSuffix();
        var cpu = GetCpuSuffix();
        var extension = AppendExtensionSuffix();

        var currentDir = $"{Path.GetDirectoryName(Assembly.GetCallingAssembly().Location)}\\".TrimEnd('\\');

        if (TryLoadFromFile(
                $"{currentDir}\\{libraryName}{extension}",
                assembly,
                searchDirectory))
        {
            return true;
        }

        if (TryLoadFromFile(
                $"{currentDir}\\bin\\{libraryName}{extension}",
                assembly,
                searchDirectory))
        {
            return true;
        }

        if (TryLoadFromFile(
                $"{currentDir}\\runtimes\\{libraryName}{extension}",
                assembly,
                searchDirectory))
        {
            return true;
        }

        if (TryLoadFromFile(
                $"{currentDir}\\{libraryName}-{os}-{cpu}{extension}",
                assembly,
                searchDirectory))
        {
            return true;
        }

        if (TryLoadFromFile(
                $"{currentDir}\\bin\\{libraryName}-{os}-{cpu}{extension}",
                assembly,
                searchDirectory))
        {
            return true;
        }

        if (TryLoadFromFile(
                $"{currentDir}\\runtimes\\{libraryName}-{os}-{cpu}{extension}",
                assembly,
                searchDirectory))
        {
            return true;
        }

        return false;
#pragma warning restore IDE0046
    }

    private static bool TryLoadFromFile(string libraryPath, Assembly assembly, DllImportSearchPath searchDirectory)
    {
        if (NativeHandles.ContainsKey(libraryPath))
        {
            return true;
        }

        if (!File.Exists(libraryPath))
        {
            return false;
        }

        var handle = NativeLibrary.Load(libraryPath, assembly, searchDirectory);

        if (handle == default)
        {
            return false;
        }

        _ = NativeHandles.TryAdd(libraryPath, handle);

        return true;
    }

    private static string GetOsSuffix()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return "win";
        }

#pragma warning disable IDE0046
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
#pragma warning restore IDE0046
        {
            return "linux";
        }

        throw new PlatformNotSupportedException("The current platform is not supported.");
    }

    private static string GetCpuSuffix()
    {
        if (RuntimeInformation.ProcessArchitecture == Architecture.X64)
        {
            return "-x64";
        }

#pragma warning disable IDE0046
        if (RuntimeInformation.ProcessArchitecture == Architecture.Arm64)
#pragma warning restore IDE0046
        {
            return "-arm64";
        }

        throw new PlatformNotSupportedException("The current platform is not supported.");
    }

    private static string AppendExtensionSuffix()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return ".dll";
        }

#pragma warning disable IDE0046
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
#pragma warning restore IDE0046
        {
            return ".so";
        }

        throw new PlatformNotSupportedException("The current platform is not supported.");
    }
}