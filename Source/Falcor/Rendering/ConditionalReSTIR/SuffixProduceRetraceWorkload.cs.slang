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

        int curPrefixLength;

    RetraceWorkloadQueue queue;

    int getNeighborCount(bool isTemporalReuse, bool finalGather)
    {
        return finalGather ? restir.subpathSettings.finalGatherSuffixCount
                               : (isTemporalReuse ? 1 : restir.subpathSettings.suffixSpatialNeighborCount);
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        if (i == -1) return pixel;
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * restir.subpathSettings.suffixSpatialReuseRadius);

        return neighborPixel;
    }

    bool isPixelPrefixLastSdValid(int offset, bool isPrevFrame, bool useScratchSuffix)
    {
        PrefixGBuffer pg;
        if (useScratchSuffix)
            pg = restir.prefixGBuffer[offset];
        else if (isPrevFrame)
            pg = prevPrefixGBuffer[offset];
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
        uint workMask = 0;
        uint centralPrefixLength = 0;
        uint neighborPrefixLengths = 0;
        bool finalGather = suffixReuseRoundId == -1;
        bool temporalReuse = restir.subpathSettings.suffixTemporalReuse && suffixReuseRoundId == 0;
        int neighborCount = getNeighborCount(temporalReuse, finalGather);

        if (all(pixel < params.frameDim))
        {
            SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() + (finalGather ? restir.finalGatherAdditionalOffset(integrationPrefixId) : suffixReuseRoundId));

            const uint centralOffset = params.getReservoirOffset(pixel);

            int centralVertexBufferOffset = (kMaxSurfaceBounces + 1) * centralOffset;

            bool isCentralPrefixValid = isPixelPrefixLastSdValid(centralOffset, false, finalGather);

            if (isCentralPrefixValid)
            {
                PathReservoir centralReservoir = {};

                if (finalGather)
                {
                    centralReservoir = restir.pathReservoirs[centralOffset];
                }
                else if (restir.subpathSettings.suffixTemporalReuse && suffixReuseRoundId == 0)
                {
                    centralReservoir = reservoirs[centralOffset];
                }
                else
                {
                    centralReservoir = prevReservoirs[centralOffset];
                }

                centralPrefixLength = centralReservoir.pathFlags.prefixLength();
                bool currentUserFlagSet = centralReservoir.pathFlags.isUserFlagSet();

                const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

                float2 motionVector = motionVectors[pixel];
                int2 reprojPixel = pixel;

                if (temporalReuse)
                {
                    reprojPixel = pixel + motionVector * params.frameDim + 0.5;
                }

                for (int i = 0; i < neighborCount; ++i)
                {
                    int2 neighborPixel;
                    if (finalGather)
                        neighborPixel = unpackSearchedPixel(foundNeighborPixels[neighborCount * centralOffset + i]);
                    else
                        neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, i - (temporalReuse ? 1 : 0));

                    if (!isValidScreenRegion(params, neighborPixel))
                    {
                        neighborValidMask[centralOffset].setValid(i, false);
                        continue;
                    }

                    int neighborOffset = params.getReservoirOffset(neighborPixel);
                    PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

                    int neighborVertexBufferOffset = (kMaxSurfaceBounces + 1) * neighborOffset;

                    bool isNeighborPrefixValid = isPixelPrefixLastSdValid(neighborOffset, temporalReuse, false);

                    if (!isNeighborPrefixValid)
                    {
                        neighborValidMask[centralOffset].setValid(i, false);
                        continue;
                    }

                    neighborValidMask[centralOffset].setValid(i, true);

                    neighborPrefixLengths |= (neighborReservoir.pathFlags.prefixLength() & 0xF) << (4 * i);

                    if (ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
                    {                        
                        int p1 = int(centralReservoir.pathFlags.rcVertexLength() > 1 + centralReservoir.pathFlags.prefixLength() &&
                                     centralReservoir.pathFlags.pathLength() >= centralReservoir.pathFlags.prefixLength() &&
                                        neighborReservoir.pathFlags.pathTreeLength() >= restir.minimumPrefixLength); //if a path have 3 trunk vertices it has pathTreeLength of 1, 
                                                                                                                         // if it doesn't have a 4th NEE vertex, it has pathLength of 0
                        numWorks += p1;
                        workMask |= p1 << (2 * i);
                        int p2 = int(neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() &&
                                     neighborReservoir.pathFlags.pathLength() >= neighborReservoir.pathFlags.prefixLength() &&
                                     centralReservoir.pathFlags.pathTreeLength() >= restir.minimumPrefixLength);
                        numWorks += p2;
                        workMask |= p2 << (2 * i + 1);
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
            int j = 0;
            for (uint i = 0; i < 2 * neighborCount; i++)
            {
                if (workMask >> i & 1)
                {
                    queue.workload.Store((dstIdx + j) * 4, pathID | (i << 24) | ((i % 2 == 0 ? (neighborPrefixLengths >> ((i/2)*4)) & 0xF : centralPrefixLength) << 28));
                    j++;
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
