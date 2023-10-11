/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/

/** Dummy compute program for reflection of the structured buffer types.

    Falcor requires reflection data to be able to create a structured buffer,
    but the programs are not yet ready to be used when we create the buffers.
    Therefore we create this dummy program to make it easier.
 */

import PathReservoir;
import PathTracer;

StructuredBuffer<PathReservoir> pathReservoirs;
StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
ParameterBlock<PathTracer> pathTracer;
StructuredBuffer<NeighborValidMask> neighborValidMask;
StructuredBuffer<PrefixGBuffer> prefixGBuffer;
StructuredBuffer<uint> foundNeighborPixels;
StructuredBuffer<PrefixReservoir> prefixReservoirs;
StructuredBuffer<float3> prefixPathReservoirs;
StructuredBuffer<uint> rcBufferOffsets;
StructuredBuffer<PrefixInfo> prefixThroughputs;
StructuredBuffer<PrefixSearchKey> prefixSearchKeys;
StructuredBuffer<PackedBoundingBox> searchPointBoundingBoxBuffer;
StructuredBuffer<float> prefixL2LengthBuffer;

void main() {}
