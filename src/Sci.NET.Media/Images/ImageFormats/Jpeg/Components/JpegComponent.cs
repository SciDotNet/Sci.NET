// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg.Components;

internal class JpegComponent : IDisposable
{
    public JpegComponent(byte id, byte samplingFactor, byte quantizationTableId)
    {
        Id = id;
        SamplingFactor = samplingFactor;
        QuantizationTableId = quantizationTableId;
    }

    public byte Id { get; }

    public int DcPredictor { get; set; }

    public int HorizontalSamplingFactor { get; }

    public int VerticalSamplingFactor { get; }

    public SystemMemoryBlock<Block8X8F> SpectralBlocks { get; set; }

    public Shape SubSamplingDevisors { get; set; }

    public byte QuantizationTableId { get; }

    public int Index { get; set; }
    
    p
}