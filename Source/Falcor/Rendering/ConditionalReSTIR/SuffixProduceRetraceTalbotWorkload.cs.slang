/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import ConditionalReSTIR;
import LoadShadingData;
import RetraceWorkloadQueue;
import Params;
import Scene.HitInfo;
import PathReservoir;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import ResamplingCommon;

// Shared memory variables.
static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;
groupshared uint gWorkloadOffset[8 /* kWarpCount */];

struct PathGenerator
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params;
    // Resources
    ConditionalReSTIR restir;  
    
    Texture1D<float2> neighborOffsets;
    Texture2D<float2> motionVectors;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;

    StructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;

    StructuredBuffer<uint> foundNeighborPixels;

    int suffixReuseRoundId;
    int integrationPrefixId;

    RetraceWorkloadQueue queue;                   ///< Work queues we append paths to.

    int getNeighborCount()
    {
        return restir.subpathSettings.finalGatherSuffixCount;
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        if (i == -1) return pixel;
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * restir.subpathSettings.suffixSpatialReuseRadius);

        return neighborPixel;
    }

    bool isPixelPrefixLastSdValid(int offset, bool useScratchSuffix)
    {
        PrefixGBuffer pg;
        if (useScratchSuffix)
            pg = restir.prefixGBuffer[offset];
        else
            pg = prefixGBuffer[offset];

        return pg.hit.isValid();
    }

    /** Entry point for path generator.
        \param[in] tileID Tile ID in x and y on screen.
        \param[in] threadIdx Thread index within the tile.
    */
    void execute(const uint2 tileID, const uint threadIdx)
    {
        const uint2 tileOffset = tileID << kScreenTileBits;            // Tile offset in pixels.
        const uint2 pixel = deinterleave_8bit(threadIdx) + tileOffset; // Assumes 16x16 tile or smaller.

        printSetPixel(pixel);

        uint numWorks = 0;
        uint3 workMask = 0;

        uint centralPrefixLength = 0;
        uint centralRcVertexLength = 0;
        uint centralPathLength = 0;
        uint centralPathTreeLength = 0;

        uint neighborPrefixLengths = 0;
        int neighborCount = getNeighborCount();

        if (all(pixel < params.frameDim))
        {
            // figure out neighbors
            SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() +
                                                            restir.finalGatherAdditionalOffset(integrationPrefixId));

            const uint centralOffset = params.getReservoirOffset(pixel);

            bool isCentralPrefixValid = isPixelPrefixLastSdValid(centralOffset, true);

            if (isCentralPrefixValid)
            {
                {
                    PathReservoir centralReservoir = restir.pathReservoirs[centralOffset];
                    centralPrefixLength = centralReservoir.pathFlags.prefixLength();
                    centralPathTreeLength = centralReservoir.pathFlags.pathTreeLength();
                    centralRcVertexLength = centralReservoir.pathFlags.rcVertexLength();
                    centralPathLength = centralReservoir.pathFlags.pathLength();
                }

                const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

                // assume at most 8 neighbors
                for (int i = 0; i < neighborCount; ++i)
                {
                    int workMaskId = i <= 3 ? 0 : 1;
                    int workMaskShiftBits = (i % 4) * 8;

                    int2 neighborPixel = unpackSearchedPixel(foundNeighborPixels[neighborCount * centralOffset + i]);

                    if (!isValidScreenRegion(params, neighborPixel))
                    {
                        neighborValidMask[centralOffset].setValid(i, false);
                        continue;
                    }

                    int neighborOffset = params.getReservoirOffset(neighborPixel);
                    bool isNeighborPrefixValid = isPixelPrefixLastSdValid(neighborOffset, false);

                    if (!isNeighborPrefixValid)
                    {
                        neighborValidMask[centralOffset].setValid(i, false);
                        continue;
                    }
                    neighborValidMask[centralOffset].setValid(i, true);

                    PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
                    int neighborRcVertexLength = neighborReservoir.pathFlags.rcVertexLength();
                    int neighborPathLength = neighborReservoir.pathFlags.pathLength();
                    int neighborPathTreeLength = neighborReservoir.pathFlags.pathTreeLength();
                    int neighborPrefixLength = neighborReservoir.pathFlags.prefixLength();

                    neighborPrefixLengths |= (neighborPrefixLength & 0xF) << (4 * i);
                    bool validForShift = neighborRcVertexLength > 1 + neighborPrefixLength &&
                                         neighborPathLength >= neighborPrefixLength;

                    if (ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid && validForShift)
                    {
                        int bufferJ = 0;
                        for (int j = 0; j < neighborCount; j++)
                        {
                            if (i == j) continue;
                            int2 dstPixel = unpackSearchedPixel(foundNeighborPixels[neighborCount * centralOffset + j]);
                            int dstOffset = params.getReservoirOffset(dstPixel);
                            if (!isValidScreenRegion(params, dstPixel) || !isPixelPrefixLastSdValid(dstOffset, false)) continue;

                            int p = int(prevReservoirs[dstOffset].pathFlags.pathTreeLength() >= restir.minimumPrefixLength);
                            numWorks += p;
                            workMask[workMaskId] |= p << (workMaskShiftBits + bufferJ);
                            bufferJ++;
                        }

                        bufferJ = neighborCount - 1;

                        {
                            int p = int(centralPathTreeLength >= restir.minimumPrefixLength);
                            numWorks += p;
                            workMask[workMaskId] |= p << (workMaskShiftBits + bufferJ);
                        }
                    }
                }

                bool centralValidForShift = centralRcVertexLength > 1 + centralPrefixLength &&
                                            centralPathLength >= centralPrefixLength;

                if (ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid && centralValidForShift)
                {
                    for (int j = 0; j < neighborCount; j++)
                    {
                        int2 dstPixel = unpackSearchedPixel(foundNeighborPixels[neighborCount * centralOffset + j]);
                        int dstOffset = params.getReservoirOffset(dstPixel);
                        if (!isValidScreenRegion(params, dstPixel) || !isPixelPrefixLastSdValid(dstOffset, false)) continue;

                        int p = int(prevReservoirs[dstOffset].pathFlags.pathTreeLength() >= restir.minimumPrefixLength);
                        numWorks += p;
                        workMask[2] |= p << j;
                    }
                }
            }
        }

        const uint warpIdx = threadIdx >> 5;

        uint numWorksInWave = WaveActiveSum(numWorks);
        if (WaveIsFirstLane())
        {
            gWorkloadOffset[warpIdx] = numWorksInWave;
        }
        GroupMemoryBarrierWithGroupSync();

        if (threadIdx < kWarpCount)
        {
            uint numWorksInWave = gWorkloadOffset[threadIdx];
            gWorkloadOffset[threadIdx] = queue.addCounter(numWorksInWave);
        }
        GroupMemoryBarrierWithGroupSync();

        if (all(pixel < params.frameDim))
        {
            uint pathID = (pixel.y << 12) | pixel.x;
            uint dstIdx = gWorkloadOffset[warpIdx] + WavePrefixSum(numWorks);

            int workId = 0;

            for (int workMaskId = 0; workMaskId < 2; workMaskId++)
            {
                int startI = 4 * workMaskId;
                int endI = workMaskId == 0 ? min(neighborCount, 4) : neighborCount;
                for (uint i = startI; i < endI; i++)
                {
                    int workMaskShiftBits = (i % 4) * 8;
                    int bufferJ = 0;
                    for (int j = 0; j < neighborCount; j++)
                    {
                        if (i == j) continue;
                        if (workMask[workMaskId] >> (workMaskShiftBits + bufferJ) & 1)
                        {
                            queue.workload.Store((dstIdx + workId) * 4, pathID | ((i * neighborCount + bufferJ) << 24));
                            queue.workloadExtra.Store((dstIdx + workId) * 4, neighborPrefixLengths >> (j * 4) & 0xF);
                            workId++;
                        }
                        bufferJ++;
                    }

                    bufferJ = neighborCount - 1;

                    if (workMask[workMaskId] >> (workMaskShiftBits + bufferJ) & 1)
                    {
                        queue.workload.Store((dstIdx + workId) * 4, pathID | ((i * neighborCount + bufferJ) << 24));
                        queue.workloadExtra.Store((dstIdx + workId) * 4, centralPrefixLength & 0xF);
                        workId++;
                    }
                }
            }

            for (uint j = 0; j < neighborCount; j++)
            {
                if (workMask[2] >> j & 1)
                {
                    queue.workload.Store((dstIdx + workId) * 4, pathID | ((neighborCount * neighborCount + j) << 24));
                    queue.workloadExtra.Store((dstIdx + workId) * 4, neighborPrefixLengths >> (j * 4) & 0xF);
                    workId++;
                }
            }
        }
    }
}

cbuffer CB
{
    PathGenerator gPathGenerator;
}

[numthreads(256 /* kScreenTileDim.x * kScreenTileDim.y */, 1, 1)]
void main(
    uint3 groupID: SV_GroupID,
    uint3 groupThreadID: SV_GroupThreadID)
{
    gPathGenerator.execute(groupID.xy, groupThreadID.x);
}
