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

struct PrefixPathRetrace
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ConditionalReSTIR restir;           


    Texture2D<float2> motionVectors;

    StructuredBuffer<PathReservoir> prevReservoirs;
    //StructuredBuffer<PathReservoir> reservoirs;

    RWStructuredBuffer<PrefixReservoir> prefixReservoirs;
    StructuredBuffer<PrefixReservoir> prevPrefixReservoirs;

    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    RWStructuredBuffer<uint> rcBufferOffsets;

    Texture2D<PackedHitInfo> vbuffer; ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;


    RWStructuredBuffer<NeighborValidMask> neighborValidMask;

    int suffixReuseRoundId;

    RetraceWorkloadQueue queue;
    StructuredBuffer<uint> foundNeighborPixels;

    RWStructuredBuffer<float> prefixTotalLengthBuffer;

    // for linear growth of M when merging reservoirs
    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX;  ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY;  ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).


#if RETRACE_SCHEDULE_TYPE == RETRACE_SCHEDULE_COMPACT
    void ReSTIR(const uint linearIndex)
    {
        // read number of paths
        uint totalPathCount = queue.counter.Load(0);
        if (linearIndex >= totalPathCount) return;
        uint pathID = queue.workload.Load(linearIndex * 4);
        int2 pixel = int2(pathID & 0xfff, pathID >> 12 & 0xfff);
        uint neighborIdx = pathID >> 24 & 0xff;
        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        printSetPixel(pixel);
        logSetPixel(pixel);

        var sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.prefixResamplingSgOffset());

        const uint centralOffset = params.getReservoirOffset(pixel);

        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel + motionVector * params.frameDim + 0.5f;
        const uint temporalOffset = params.getReservoirOffset(prevPixel);

        PathReservoir reservoir;
        uint componentType = 0;
        if (neighborIdx % 2)
        {
            reservoir = prevReservoirs[temporalOffset];
            reservoir.pathFlags.insertPathTreeLength(1);
            componentType = prevPrefixReservoirs[temporalOffset].componentType();
        }
        else
        {
            reservoir = restir.pathReservoirs[centralOffset];
            reservoir.pathFlags.insertPathTreeLength(0);
            componentType = prefixReservoirs[centralOffset].componentType();
        }

        HitInfo primaryHit;

        ShadingData primarySd = getPixelShadingDataGeneral(params, vbuffer, temporalVbuffer, neighborIdx % 2 ? pixel : prevPixel, primaryHit, neighborIdx % 2 == 0, pcf);

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);

        ReSTIRPathFlags tempFlag = reservoir.pathFlags;
        tempFlag.insertRcVertexLength(tempFlag.prefixLength());
        tempFlag.insertPathLength(restir.kMaximumRcLength);
        tempFlag.insertBSDFComponentType(componentType, true);

        float prefixPartTotalLength;
        float3 thp = TraceReplayedPath(primaryHit,

                                       primarySd,

                                       tempFlag,
                                       reservoir.initRandomSeed, reservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, neighborIdx % 2 == 0, prefixPartTotalLength, false, true);
        float prefixTotalLength = 0;
        if ((neighborIdx % 2 ) && restir.subpathSettings.knnSearchAdaptiveRadiusType == (uint)ConditionalReSTIR::KNNAdaptiveRadiusType::RayCone)
        {
            prefixTotalLength = length(gScene.camera.data.posW - primarySd.posW);
            const int pixelId = pixel.y * params.frameDim.x + pixel.x;
            prefixTotalLengthBuffer[pixelId] = prefixTotalLength + prefixPartTotalLength;
        }

        rcBufferOffsets[2 * centralOffset + neighborIdx] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    }
#else
    bool ReSTIR(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restir.sgCount() * params.seed + restir.prefixResamplingSgOffset());
        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = restir.pathReservoirs[centralOffset];

        HitInfo centralPrimaryHit;
        ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);

        if (!centralPrimaryHit.isValid()) return false;

        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel + motionVector * params.frameDim + 0.5f;
        int prevOffset = params.getReservoirOffset(prevPixel);

        if (!isValidScreenRegion(params, prevPixel)) return false;

        HitInfo temporalPrimaryHit;
        ShadingData temporalPrimarySd = getPixelTemporalShadingData(params, temporalVbuffer, prevPixel, temporalPrimaryHit, pcf);
        if (!temporalPrimaryHit.isValid() || !isValidGeometry(centralPrimarySd, temporalPrimarySd)) return false;

        PathReservoir temporalReservoir = prevReservoirs[prevOffset];

        // talbot MIS
        // compute mis weight for current pixel

        const int startReplayPrefixLength = 1;

        
        if (centralReservoir.pathFlags.prefixLength() > startReplayPrefixLength &&
            centralReservoir.pathFlags.pathTreeLength() >= centralReservoir.pathFlags.prefixLength())
        {
            HitInfo rcPrevHit = HitInfo();
            float3 rcPrevWo = float3(0.f);
            float3 thp = 1.f;
            float dummy;

            ReSTIRPathFlags tempFlag = centralReservoir.pathFlags;
            tempFlag.insertRcVertexLength(tempFlag.prefixLength());
            tempFlag.insertPathLength(restir.kMaximumRcLength);
            tempFlag.insertBSDFComponentType(prefixReservoirs[centralOffset].componentType(), true);

            thp = TraceReplayedPath(temporalPrimaryHit,

                                    temporalPrimarySd,

                                    tempFlag,
                                    centralReservoir.initRandomSeed, centralReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, true, dummy, false, true);

            rcBufferOffsets[2 * centralOffset] = 2 * centralOffset;
            reconnectionDataBuffer[2 * centralOffset] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
        }

        if (temporalReservoir.pathFlags.prefixLength() > startReplayPrefixLength &&
            temporalReservoir.pathFlags.pathTreeLength() >= temporalReservoir.pathFlags.prefixLength())
        {
            HitInfo rcPrevHit = HitInfo();
            float3 rcPrevWo = float3(0.f);
            float3 thp = 1.f;
            float prefixPartTotalLength = 0.f;

            ReSTIRPathFlags tempFlag = temporalReservoir.pathFlags;
            tempFlag.insertRcVertexLength(tempFlag.prefixLength());
            tempFlag.insertPathLength(restir.kMaximumRcLength);
            tempFlag.insertBSDFComponentType(prevPrefixReservoirs[prevOffset].componentType(), true);

            thp = TraceReplayedPath(centralPrimaryHit,

                                    centralPrimarySd,

                                    tempFlag,
                                    temporalReservoir.initRandomSeed, temporalReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, prefixPartTotalLength, false, true);
            const int pixelId = pixel.y * params.frameDim.x + pixel.x;

            if (restir.subpathSettings.knnSearchAdaptiveRadiusType == (uint)ConditionalReSTIR::KNNAdaptiveRadiusType::RayCone)
            {
                float prefixTotalLength = length(gScene.camera.data.posW - centralPrimarySd.posW);
                prefixTotalLengthBuffer[pixelId] = prefixTotalLength + prefixPartTotalLength;
            }
            rcBufferOffsets[2 * centralOffset + 1] = 2 * centralOffset + 1;
            reconnectionDataBuffer[2 * centralOffset + 1] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
        }
        return true;
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

        bool isValid = ReSTIR(pixel);
        const uint centralOffset = params.getReservoirOffset(pixel);
        neighborValidMask[centralOffset].setValid(0, isValid);
    }
#endif
};

cbuffer CB
{
    PrefixPathRetrace gPrefixPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gPrefixPathRetrace.execute(dispatchThreadId.x);
#else
    gPrefixPathRetrace.execute(dispatchThreadId.xy);
#endif
}
