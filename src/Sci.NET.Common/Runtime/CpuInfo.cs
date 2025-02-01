// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Management;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Sci.NET.Common.Runtime;

/// <summary>
/// Provides information about the CPU.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
public static class CpuInfo
{
    static CpuInfo()
    {
        var vendor = "Unknown Vendor";
        var model = "Unknown Model";
        var clockSpeed = 0.0d;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                (vendor, model, clockSpeed) = GetWindowsInfo();
            }
            catch
            {
                // ignored
            }
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            try
            {
                (vendor, model, clockSpeed) = GetLinuxOsxInfo();
            }
            catch
            {
                // ignored
            }
        }

        Vendor = vendor;
        Model = model;
        Cores = Environment.ProcessorCount;
        ClockSpeed = clockSpeed / 1000.0d;
    }

    /// <summary>
    /// Gets the vendor of the CPU.
    /// </summary>
    public static string Vendor { get; }

    /// <summary>
    /// Gets the model of the CPU.
    /// </summary>
    public static string Model { get; }

    /// <summary>
    /// Gets the number of cores of the CPU.
    /// </summary>
    public static int Cores { get; }

    /// <summary>
    /// Gets the clock speed of the CPU in GHz.
    /// </summary>
    public static double ClockSpeed { get; }

    /// <summary>
    /// Gets a string representation of the CPU information.
    /// </summary>
    /// <returns>A string containing the CPU info.</returns>
#pragma warning disable CA1024
    public static string GetInfoString()
#pragma warning restore CA1024
    {
        return $"CPU: {Vendor} {Model} ({Cores} cores @ {ClockSpeed} GHz)";
    }

    [SupportedOSPlatformGuard("windows")]
    private static (string Vendor, string Model, double ClockSpeed) GetWindowsInfo()
    {
#pragma warning disable CA1416
        using var searcher = new ManagementObjectSearcher("root\\CIMV2", "SELECT * FROM Win32_Processor");
        var vendor = string.Empty;
        var model = string.Empty;
        var clockSpeed = 0.0d;

        foreach (var queryObj in searcher.Get())
        {
            vendor = queryObj["Manufacturer"].ToString()?.Trim(' ') ?? string.Empty;
            model = queryObj["Name"].ToString()?.Trim(' ') ?? string.Empty;

            clockSpeed = double.Parse(
                queryObj["MaxClockSpeed"].ToString() ?? "0.0",
                CultureInfo.CurrentCulture);
            break;
        }
#pragma warning restore CA1416

        return (vendor, model, clockSpeed);
    }

    private static (string Vendor, string Model, double ClockSpeed) GetLinuxOsxInfo()
    {
        var vendor = string.Empty;
        var model = string.Empty;
        var clockSpeed = 0.0d;

        var cpuInfo = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "/usr/bin/env",
            Arguments = "cat /proc/cpuinfo",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = System.Diagnostics.Process.Start(cpuInfo);

        while (process is { StandardOutput.EndOfStream: false })
        {
            var line = (process.StandardOutput.ReadLine() ?? string.Empty).Trim();

            if (line.StartsWith("vendor_id", StringComparison.InvariantCulture))
            {
                vendor = line.Split(':')[1].Trim();
            }
            else if (line.StartsWith("model name", StringComparison.InvariantCulture))
            {
                model = line.Split(':')[1].Trim();
            }
            else if (line.StartsWith("cpu MHz", StringComparison.InvariantCulture))
            {
                clockSpeed = double.Parse(line.Split(':')[1].Trim(), CultureInfo.CurrentCulture);
            }
        }

        return (vendor, model, clockSpeed);
    }
}