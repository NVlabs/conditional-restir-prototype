/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
import Params;
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingData;
import HybridShiftTraceRays;
import RetraceWorkloadQueue;
import ResamplingCommon;
#include "RetraceScheduleDefinition.slangh"


struct SuffixPathRetrace
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ConditionalReSTIR restir;             ///< Settings for ReSTIR. Only valid if kConditionalReSTIR == true.


    Texture1D<float2> neighborOffsets;
    Texture2D<float2> motionVectors;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;

    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    RWStructuredBuffer<uint> rcBufferOffsets;

    StructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;

    RWStructuredBuffer<NeighborValidMask> neighborValidMask;

    int suffixReuseRoundId;
    int integrationPrefixId;

    RetraceWorkloadQueue queue;
    StructuredBuffer<uint> foundNeighborPixels;

    // for linear growth of M when merging reservoirs

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

    ShadingData getRcPrevShadingData(HitInfo rcPrevHit, float3 rcPrevWo)
    {
        ShadingData sd = {};
        if (rcPrevHit.isValid())
        {
            sd = loadShadingData(rcPrevHit, 0.f, -rcPrevWo, false, gPathTracer.createTextureSampler(), false);
        }
        return sd;
    }


    ShadingData getPixelPrefixLastSd(int2 pixel, bool useScratchSuffix, int offset, inout HitInfo prefixLastHit)
    
    {
        // temporal vertices?
        PrefixGBuffer sfPrevVertex;
        if (useScratchSuffix)
            sfPrevVertex = restir.prefixGBuffer[offset];
        else
            sfPrevVertex = prefixGBuffer[offset];

        if (!sfPrevVertex.hit.isValid()) return {};

        prefixLastHit = sfPrevVertex.hit;
        return getRcPrevShadingData(prefixLastHit, sfPrevVertex.wo);
        
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

#if RETRACE_SCHEDULE_TYPE == RETRACE_SCHEDULE_COMPACT
    void ReSTIR(const uint linearIndex)
    {
        // read number of paths
        uint totalPathCount = queue.counter.Load(0);
        if (linearIndex >= totalPathCount) return;
        uint pathID = queue.workload.Load(linearIndex * 4);
        int2 pixel = int2(pathID & 0xfff, pathID >> 12 & 0xfff);
        uint neighborIdx = pathID >> 24 & 0xFF;
        int dstPrefixLength = queue.workloadExtra.Load(linearIndex * 4);

        printSetPixel(pixel);
        logSetPixel(pixel);

        SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() + 
                                                restir.finalGatherAdditionalOffset(integrationPrefixId));

        const uint centralOffset = params.getReservoirOffset(pixel);

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        
        PathReservoir reservoir;

        int neighborCount = getNeighborCount();

        int srcId = neighborIdx / neighborCount;
        srcId = srcId == neighborCount ? -1 : srcId;

        int2 srcPixel = pixel;

        if (srcId >= 0)
        {
            srcPixel = unpackSearchedPixel(foundNeighborPixels[neighborCount * centralOffset + srcId]);
        }

        uint srcOffset = params.getReservoirOffset(srcPixel);

        if (srcId == -1)
        {
            reservoir = restir.pathReservoirs[centralOffset];
        }
        else
        {
            reservoir = prevReservoirs[srcOffset];
        }

        int dstId = neighborIdx % neighborCount;
        if (srcId >= 0 && dstId >= srcId) dstId++;
        if (dstId == neighborCount) dstId = -1;

        int2 dstPixel = pixel;
        if (dstId >= 0)
        {
            dstPixel = unpackSearchedPixel(foundNeighborPixels[neighborCount * centralOffset + dstId]);
        }

        uint dstOffset = params.getReservoirOffset(dstPixel);

        HitInfo PrefixLastHit;

        ShadingData PrefixLastSd = getPixelPrefixLastSd(dstPixel, dstId == -1, dstOffset, PrefixLastHit);


        // modify flags
        ReSTIRPathFlags tempFlag = reservoir.pathFlags;
        if (restir.subpathSettings.adaptivePrefixLength)
        {
            int lengthChange = dstPrefixLength - tempFlag.prefixLength();
            tempFlag.insertPrefixLength(dstPrefixLength);
            if (tempFlag.rcVertexLength() != restir.kMaximumRcLength)
                tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
            tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
        }

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float3 thp = 0.f;
        float dummy;

        if (tempFlag.pathLength() <= kMaxSurfaceBounces)
        {
            thp = TraceReplayedPath(PrefixLastHit,

                                    PrefixLastSd,

                                    tempFlag,
                                    reservoir.initRandomSeed, reservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy, true);
        }
        rcBufferOffsets[(neighborCount + 1) * neighborCount * centralOffset + neighborIdx] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    }
#else
    void ReSTIR(const uint2 pixel)
    {
        SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() + 
                                                            restir.finalGatherAdditionalOffset(integrationPrefixId));

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = restir.pathReservoirs[centralOffset];
        HitInfo centralPrefixLastHit;

        ShadingData centralPrefixLastSd = getPixelPrefixLastSd(pixel, true, centralOffset, centralPrefixLastHit);
        if (all(centralPrefixLastSd.V == 0.f)) return;


        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        const int numNeighbors = restir.subpathSettings.finalGatherSuffixCount;

        for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
        {
            int2 neighborPixel = unpackSearchedPixel(foundNeighborPixels[numNeighbors * centralOffset + neighborId]);

            if (!isValidScreenRegion(params, neighborPixel))
            {
                neighborValidMask[centralOffset].setValid(neighborId, false);
                continue;
            }

            int neighborOffset = params.getReservoirOffset(neighborPixel);

            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
            if (!isPixelPrefixLastSdValid(neighborOffset, false))
            {
                neighborValidMask[centralOffset].setValid(neighborId, false);
                continue;
            }

            // talbot MIS
            int bufferJ = 0;
            for (int j = 0; j < numNeighbors; j++)
            {
                if (neighborId == j) continue;

                int2 dstPixel = unpackSearchedPixel(foundNeighborPixels[numNeighbors * centralOffset + j]);
                if (!isValidScreenRegion(params, dstPixel)) continue;

                int dstOffset = params.getReservoirOffset(dstPixel);
                if (!isPixelPrefixLastSdValid(dstOffset, false)) continue;

                PathReservoir dstReservoir = prevReservoirs[dstOffset];

                HitInfo dstPrefixLastHit;

                ShadingData dstPrefixLastSd = getPixelPrefixLastSd(dstPixel, false, dstOffset, dstPrefixLastHit);
                if (all(dstPrefixLastSd.V == 0.f)) continue;


                if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() &&
                    neighborReservoir.pathFlags.pathLength() >= neighborReservoir.pathFlags.prefixLength() &&
                    dstReservoir.pathFlags.pathTreeLength() >= restir.minimumPrefixLength)
                {
                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 thp = 0.f;
                    float dummy;
                    ReSTIRPathFlags tempFlag = neighborReservoir.pathFlags;
                    if (restir.subpathSettings.adaptivePrefixLength)
                    {
                        int dstPrefixLength = dstReservoir.pathFlags.prefixLength();
                        int lengthChange = dstPrefixLength - neighborReservoir.pathFlags.prefixLength();
                        if (dstReservoir.pathFlags.pathLength() + lengthChange > kMaxSurfaceBounces)
                            tempFlag.insertPathLength(0); // nullify the sample
                        tempFlag.insertPrefixLength(dstPrefixLength);
                        if (tempFlag.rcVertexLength() != restir.kMaximumRcLength)
                            tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
                        tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
                    }

                    if (tempFlag.pathLength() <= kMaxSurfaceBounces)
                    {
                        thp = TraceReplayedPath(dstPrefixLastHit,

                                                dstPrefixLastSd,

                                                tempFlag,
                                                neighborReservoir.initRandomSeed, neighborReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy, true);
                    }

                    rcBufferOffsets[numNeighbors * (numNeighbors + 1) * centralOffset + neighborId * numNeighbors + bufferJ] = 
                                                    numNeighbors * (numNeighbors + 1) * centralOffset + neighborId * numNeighbors + bufferJ;
                    reconnectionDataBuffer[numNeighbors * (numNeighbors + 1) * centralOffset + neighborId * numNeighbors + bufferJ] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
                }

                // compute MIS weight
                bufferJ++;
            }

            // shift to central
            if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() &&
                neighborReservoir.pathFlags.pathLength() >= neighborReservoir.pathFlags.prefixLength() &&
                centralReservoir.pathFlags.pathTreeLength() >= restir.minimumPrefixLength)
            {
                HitInfo rcPrevHit = HitInfo();
                float3 rcPrevWo = float3(0.f);
                float3 thp = 0.f;
                float dummy;
                ReSTIRPathFlags tempFlag = neighborReservoir.pathFlags;
                if (restir.subpathSettings.adaptivePrefixLength)
                {
                    int dstPrefixLength = centralReservoir.pathFlags.prefixLength();
                    int lengthChange = dstPrefixLength - neighborReservoir.pathFlags.prefixLength();
                    if (centralReservoir.pathFlags.pathLength() + lengthChange > kMaxSurfaceBounces)
                        tempFlag.insertPathLength(0); // nullify the sample
                    tempFlag.insertPrefixLength(dstPrefixLength);
                    if (tempFlag.rcVertexLength() != restir.kMaximumRcLength)
                        tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
                    tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
                }

                if (tempFlag.pathLength() <= kMaxSurfaceBounces)
                {
                    thp = TraceReplayedPath(centralPrefixLastHit,

                                            centralPrefixLastSd,

                                            tempFlag,
                                            neighborReservoir.initRandomSeed, neighborReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy, true);
                }
                rcBufferOffsets[numNeighbors * (numNeighbors + 1) * centralOffset + neighborId * numNeighbors + numNeighbors - 1] = 
                                                        numNeighbors * (numNeighbors + 1) * centralOffset + neighborId * numNeighbors + numNeighbors - 1;
                reconnectionDataBuffer[numNeighbors * (numNeighbors + 1) * centralOffset + neighborId * numNeighbors + numNeighbors - 1] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
            }
        }

        // shift central to the neighbors
        // talbot MIS
        for (int j = 0; j < numNeighbors; j++)
        {
            int2 dstPixel = unpackSearchedPixel(foundNeighborPixels[numNeighbors * centralOffset + j]);

            if (!isValidScreenRegion(params, dstPixel)) continue;
            int dstOffset = params.getReservoirOffset(dstPixel);
            if (!isPixelPrefixLastSdValid(dstOffset, false)) continue;

            PathReservoir dstReservoir = prevReservoirs[dstOffset];

            HitInfo dstPrefixLastHit;

            ShadingData dstPrefixLastSd = getPixelPrefixLastSd(dstPixel, false, dstOffset, dstPrefixLastHit);
            if (all(dstPrefixLastSd.V == 0.f)) continue;


            if (centralReservoir.pathFlags.rcVertexLength() > 1 + centralReservoir.pathFlags.prefixLength() &&
                centralReservoir.pathFlags.pathLength() >= centralReservoir.pathFlags.prefixLength() &&
                dstReservoir.pathFlags.pathTreeLength() >= restir.minimumPrefixLength)
            {
                HitInfo rcPrevHit = HitInfo();
                float3 rcPrevWo = float3(0.f);
                float3 thp = 0.f;
                float dummy;
                ReSTIRPathFlags tempFlag = centralReservoir.pathFlags;
                if (restir.subpathSettings.adaptivePrefixLength)
                {
                    int dstPrefixLength = dstReservoir.pathFlags.prefixLength();
                    int lengthChange = dstPrefixLength - centralReservoir.pathFlags.prefixLength();
                    if (dstReservoir.pathFlags.pathLength() + lengthChange > kMaxSurfaceBounces)
                        tempFlag.insertPathLength(0); // nullify the sample
                    tempFlag.insertPrefixLength(dstPrefixLength);
                    if (tempFlag.rcVertexLength() != restir.kMaximumRcLength)
                        tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
                    tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
                }

                if (tempFlag.pathLength() <= kMaxSurfaceBounces)
                {
                    thp = TraceReplayedPath(dstPrefixLastHit,

                                            dstPrefixLastSd,

                                            tempFlag,
                                            centralReservoir.initRandomSeed, centralReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy, true);
                }

                rcBufferOffsets[numNeighbors * (numNeighbors + 1) * centralOffset + numNeighbors * numNeighbors + j] = numNeighbors * (numNeighbors + 1) * centralOffset + numNeighbors * numNeighbors + j;
                reconnectionDataBuffer[numNeighbors * (numNeighbors + 1) * centralOffset + numNeighbors * numNeighbors + j] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
            }
        }
    }
#endif
    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    void execute(const uint index)
    {
        ReSTIR(index);
    }
#else
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
#endif
};

cbuffer CB
{
    SuffixPathRetrace gSuffixPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gSuffixPathRetrace.execute(dispatchThreadId.x);
#else
    gSuffixPathRetrace.execute(dispatchThreadId.xy);
#endif
}
