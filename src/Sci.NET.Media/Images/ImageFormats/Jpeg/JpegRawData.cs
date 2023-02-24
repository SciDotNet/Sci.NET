// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Media.Images.ImageFormats.Jpeg;

internal class JpegRawData
{
    private static readonly byte[] SupportedBitsPerComponent =
    {
        8, 12
    };

    private bool _hasExif;
    private byte[] _exifData;
    private bool _hasIptc;
    private byte[] _iptcData;
    private bool _hasXmp;
    private byte[] _xmpData;
    private bool _hasIcc;
    private byte[] _iccData;
    private bool _hasAdobe;
    private byte[] _adobeData;
}