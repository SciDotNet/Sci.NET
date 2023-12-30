// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Datasets.Storage.Compression;

/// <summary>
/// The compression algorithm.
/// </summary>
[PublicAPI]
public enum DatasetCompressionType
{
    /// <summary>
    /// No compression.
    /// </summary>
    None = 0,

    /// <summary>
    /// GZip compression.
    /// </summary>
    GZip = 1,
}