// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Media.Images.ImageFormats.Jpeg;

/// <summary>
/// The JPEG markers.
/// </summary>
public enum JpegMarkers : byte
{
    /// <summary>
    /// Prefix for all JPEG markers.
    /// </summary>
    XFF = 0xFF,

    /// <summary>
    /// Start of Image.
    /// </summary>
    SOI = 0xD8,

    /// <summary>
    /// End of Image.
    /// </summary>
    EOI = 0xD9,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP0 = 0xE0,

    /// <summary>
    /// Application specific marker for the metadata in the JPEG format.
    /// </summary>
    APP1 = 0xE1,

    /// <summary>
    /// Application specific marker for the ICC profile in the JPEG format.
    /// </summary>
    APP2 = 0xE2,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP3 = 0xE3,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP4 = 0xE4,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP5 = 0xE5,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP6 = 0xE6,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP7 = 0xE7,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP8 = 0xE8,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP9 = 0xE9,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP10 = 0xEA,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP11 = 0xEB,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP12 = 0xEC,

    /// <summary>
    /// Application specific marker for the JPEG format.
    /// </summary>
    APP13 = 0xED,

    /// <summary>
    /// Application specific marker for Adobe encoding information for DCT filters.
    /// </summary>
    APP14 = 0xEE,

    /// <summary>
    /// Application specific marker for Graphic Converter to store JPEG quality.
    /// </summary>
    APP15 = 0xEF,

    /// <summary>
    /// Define arithmetic coding.
    /// </summary>
    DAC = 0xCC,
    COM = 0xFE,
    DQT = 0xDB,
    SOF0 = 0xC0,
    SOF1 = 0xC1,
    SOF2 = 0xC2,
    SOF3 = 0xC3,
    SOF5 = 0xC5,
    SOF6 = 0xC6,
    SOF7 = 0xC7,
    SOF9 = 0xC9,
    SOF10 = 0xCA,
    SOF11 = 0xCB,
    SOF13 = 0xCD,
    SOF14 = 0xCE,
    SOF15 = 0xCF,
    DHT = 0xC4,
    DRI = 0xDD,
    SOS = 0xDA,
    RST0 = 0xD0,
    RST1 = 0xD1,
    RST2 = 0xD2,
    RST3 = 0xD3,
    RST4 = 0xD4,
    RST5 = 0xD5,
    RST6 = 0xD6,
    RST7 = 0xD7,
}