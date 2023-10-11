
/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.RaytracingInline;
import ConditionalReSTIR;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;
import LoadShadingData;
import Params;
import SubpathReuse;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Shift;
import ResamplingCommon;

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct SuffixResampling
{
    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ConditionalReSTIR restir;

    Texture2D<PackedHitInfo> vbuffer;         ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer; ///< Fullscreen V-buffer for the primary hits.

    Texture2D<float2> motionVectors;    
    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;

    StructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;

    Texture1D<float2> neighborOffsets;

    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> rcBufferOffsets;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;
    RWTexture2D<float4> outColor;
    StructuredBuffer<uint> foundNeighborPixels;

    int suffixReuseRoundId;
    int integrationPrefixId;
    bool hasCanonicalSuffix;

    StructuredBuffer<PrefixReservoir> prefixReservoirs;
    
    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX;  ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY;  ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    int curPrefixLength;

    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * restir.subpathSettings.suffixSpatialReuseRadius);

        return neighborPixel;
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int radius, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * radius);

        return neighborPixel;
    }

    ReconnectionData fillReconnectionPartialShiftRcDataPrefix(int offset)
    {
        PrefixGBuffer pg = prefixGBuffer[offset];
        return ReconnectionData(pg.hit, pg.wo, float3(1.f));
    }

    ReconnectionData fillReconnectionPartialShiftRcDataPrefixPrevFrame(int offset)
    {
        PrefixGBuffer pg = prevPrefixGBuffer[offset];
        return ReconnectionData(pg.hit, pg.wo, float3(1.f));
    }

    bool isPixelPrefixLastSdValid(int offset)
    {
        PrefixGBuffer pg = prefixGBuffer[offset];
        return pg.hit.isValid();
    }

    bool isPixelNewPrefixLastSdValid(int offset)
    {
        PrefixGBuffer pg = restir.prefixGBuffer[offset];
        return pg.hit.isValid();
    }

    void adjustNeighborWeight(inout float M, int numNeighbors)
    {
        if (restir.subpathSettings.useMMIS) return;
        M = min(restir.subpathSettings.temporalHistoryLength, M);
        M *= restir.subpathSettings.nonCanonicalWeightMultiplier;
    }

    void runGatherSuffixes(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);

        SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() 
                                                            + restir.finalGatherAdditionalOffset(integrationPrefixId));

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool primaryHitValid = isPrimaryHitValid(vbuffer, pixel, offset);
        if (!primaryHitValid)
        {
            return;
        }

        bool centralPrefixValid = isPixelNewPrefixLastSdValid(offset);

        ShadingData primarySd = getPixelShadingData(params, vbuffer, pixel);
        float3 primaryPosW = primarySd.posW;

        PathReservoir subpathReservoir;

        float3 res = 0.f;

        int currentPathTreeLength = 0;
        int currentRcVertexLength = 0;
        int prefixLength = 0;

        bool hasValidTemporalNeighbor = false;

        if (centralPrefixValid)
        {
            PathReservoir currentReservoir = restir.pathReservoirs[offset];
            currentPathTreeLength = currentReservoir.pathFlags.pathTreeLength();
            prefixLength = currentReservoir.pathFlags.prefixLength();
            currentRcVertexLength = currentReservoir.pathFlags.rcVertexLength();

            const int numNeighbors = restir.subpathSettings.finalGatherSuffixCount;

            float canonicalMisWeights = 0.f;
            int validNeighbors = 0;
            const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

            for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
            {
                int2 neighborPixel = unpackSearchedPixel(foundNeighborPixels[numNeighbors * offset + neighborId]);
                int neighborOffset = params.getReservoirOffset(neighborPixel);

                PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
                if (!neighborValidMask[offset].isValid(neighborId)) continue;
                hasValidTemporalNeighbor = true;

                validNeighbors++;

                adjustNeighborWeight(neighborReservoir.M, numNeighbors);

                if (restir.subpathSettings.useTalbotMISForGather)
                {
                    // talbot MIS
                    float talbotMISWeightSum = 0.f;
                    int bufferJ = 0;
                    for (int j = 0; j < numNeighbors; j++)
                    {
                        if (neighborId == j) continue;

                        int2 dstPixel = unpackSearchedPixel(foundNeighborPixels[numNeighbors * offset + j]);
                        int dstOffset = params.getReservoirOffset(dstPixel);

                        if (!neighborValidMask[offset].isValid(j)) continue;

                        ReconnectionData rcDataShift;
                        if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                        && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
                            rcDataShift = reconnectionDataBuffer[rcBufferOffsets[numNeighbors * (numNeighbors+1) * offset + neighborId*numNeighbors+bufferJ]];
                        else
                            rcDataShift = fillReconnectionPartialShiftRcDataPrefix(dstOffset);

                        PathReservoir dstReservoir = prevReservoirs[dstOffset];
                        adjustNeighborWeight(dstReservoir.M, numNeighbors);
                        talbotMISWeightSum += computeTalbotMISWeightTerm(neighborReservoir, dstReservoir, rcDataShift);
                        bufferJ++;
                    }

                    ReconnectionData rcDataShift;
                    if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                    && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
                        rcDataShift = reconnectionDataBuffer[rcBufferOffsets[numNeighbors * (numNeighbors+1) * offset + neighborId*numNeighbors+numNeighbors-1]];
                    else
                    {
                        PrefixGBuffer pg = restir.prefixGBuffer[offset];
                        rcDataShift = ReconnectionData(pg.hit, pg.wo, float3(1.f));
                    }
                    res += integrateWithNeighborReservoirSampleTalbotMIS(currentReservoir,
                                                                    neighborReservoir,
                                                                    rcDataShift,
                                                                    talbotMISWeightSum);
                }
                else
                {
                    ReconnectionData rcDataBackShift;
                    if (currentReservoir.pathFlags.rcVertexLength() > 1 + currentReservoir.pathFlags.prefixLength() 
                    && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
                    {
                        rcDataBackShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId]];
                    }
                    else
                    {
                        rcDataBackShift = fillReconnectionPartialShiftRcDataPrefix(neighborOffset);
                    }

                    ReconnectionData rcDataShift;
                    if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                    && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
                    {
                        rcDataShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId + 1]];
                    }
                    else
                    {
                        PrefixGBuffer pg = restir.prefixGBuffer[offset];
                        rcDataShift = ReconnectionData(pg.hit, pg.wo, float3(1.f));
                    }

                    res += integrateWithNeighborReservoirSample(currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                            1.f, canonicalMisWeights);
                }
            }

            canonicalMisWeights = hasValidTemporalNeighbor ? canonicalMisWeights : 1.f;
            validNeighbors = hasValidTemporalNeighbor ? validNeighbors : 1;

            if (hasCanonicalSuffix && !restir.subpathSettings.disableCanonical)
            {
                if (restir.subpathSettings.useTalbotMISForGather)
                {
                    // talbot MIS
                    float talbotMISWeightSum = 0.f;
                    for (int j = 0; j < numNeighbors; j++)
                    {
                        int2 dstPixel = unpackSearchedPixel(foundNeighborPixels[numNeighbors * offset + j]);
                        int dstOffset = params.getReservoirOffset(dstPixel);
                        if (!neighborValidMask[offset].isValid(j)) continue;

                        PathReservoir dstReservoir = prevReservoirs[dstOffset];
                        ReconnectionData rcDataShift;
                        if (currentRcVertexLength > 1 + prefixLength 
                        && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
                            rcDataShift = reconnectionDataBuffer[rcBufferOffsets[numNeighbors * (numNeighbors+1) * offset + numNeighbors*numNeighbors+j]];
                        else
                            rcDataShift = fillReconnectionPartialShiftRcDataPrefix(dstOffset);

                        adjustNeighborWeight(dstReservoir.M, numNeighbors);
                        talbotMISWeightSum += computeTalbotMISWeightTerm(currentReservoir, dstReservoir, rcDataShift);
                    }

                    canonicalMisWeights = currentReservoir.M * PathRisState::toScalar(currentReservoir.integrand) /
                                          (currentReservoir.M * PathRisState::toScalar(currentReservoir.integrand) + talbotMISWeightSum);
                    canonicalMisWeights = (isinf(canonicalMisWeights) || isnan(canonicalMisWeights)) ? 0.f : canonicalMisWeights;
                }
                res += currentReservoir.integrand * currentReservoir.weight * canonicalMisWeights * (!restir.subpathSettings.generateCanonicalSuffixForEachPrefix ? 
                                        restir.subpathSettings.numIntegrationPrefixes : 1.f);
            }

            if (!restir.subpathSettings.useTalbotMISForGather)
                res /= validNeighbors;
            res = any(isnan(res) || isinf(res))
                        ? 0.f
                        : max(0.f, res);
        }

        int vertexBufferOffset = (kMaxSurfaceBounces + 1) * offset;

        prefixLength = !centralPrefixValid ? currentPathTreeLength + 1 : prefixLength;  
        
        // multiply with prefix throughput
        if (centralPrefixValid)
        {
            if (prefixLength > 0)
            { 
                float3 prefixThroughput = restir.prefixThroughputs[offset].thp;
                res *= prefixThroughput;
            }
        }
        else
        {
            subpathReservoir.weight = 0;
            subpathReservoir.M = 1;
        }

        subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                      ? 0.f
                                      : max(0.f, subpathReservoir.weight);
        subpathReservoir.resetM();

        subpathReservoir.pathFlags.insertPathTreeLength(currentPathTreeLength);
        subpathReservoir.pathFlags.insertPrefixLength(prefixLength);

        // merge with reservoir

        if (restir.subpathSettings.numIntegrationPrefixes > 1)
        {
            // add prefix contribution
            if (integrationPrefixId == restir.subpathSettings.numIntegrationPrefixes - 1)
            {
                res += restir.prefixPathReservoirs[offset];
            }
            outColor[pixel] += float4(res / restir.subpathSettings.numIntegrationPrefixes, 0.f);
        }
        else
        {
            // merge with preifx reservoir
            if (integrationPrefixId == restir.subpathSettings.numIntegrationPrefixes - 1)
            {
                res += restir.prefixPathReservoirs[offset];
            }

            outColor[pixel] += float4(res, 0.f);
        }
    }

    void runTemporal(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool centralPrefixValid = isPixelPrefixLastSdValid(offset);
        if (!centralPrefixValid) return;

        SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() + suffixReuseRoundId);

        PathReservoir subpathReservoir;
        subpathReservoir.init();

        int vertexBufferOffset = (kMaxSurfaceBounces + 1) * offset;

        float3 color = 0.f;

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        PathReservoir currentReservoir = reservoirs[offset];

        int currentPathTreeLength = currentReservoir.pathFlags.pathTreeLength();
        int currentPrefixLength = currentReservoir.pathFlags.prefixLength();
        bool currentUserFlagSet = currentReservoir.pathFlags.isUserFlagSet();

        static const bool doTemporalUpdate = restir.kTemporalUpdateForDynamicScene;

        bool hasValidTemporalNeighbor = false;

        int numNeighbors = 1;

        float2 motionVector = motionVectors[pixel];
        int2 reprojPixel = pixel + motionVector * params.frameDim + 0.5;
        float canonicalMisWeights = 0.f;

        float3 currentPrefixDir = prefixGBuffer[offset].wo;

        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        ReconnectionData rcDataCenter = fillReconnectionPartialShiftRcDataPrefix(offset);

        for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
        {
            int2 neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, neighborId - 1);

            if (!neighborValidMask[offset].isValid(neighborId)) continue;

            int neighborOffset = params.getReservoirOffset(neighborPixel);

            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            neighborReservoir.M = min(restir.subpathSettings.temporalHistoryLength, neighborReservoir.M);
            hasValidTemporalNeighbor = true;

            ReconnectionData rcDataBackShift;

            if (currentReservoir.pathFlags.rcVertexLength() > 1 + currentReservoir.pathFlags.prefixLength()
                && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
            {
                rcDataBackShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId]];
            }
            else
            {
                rcDataBackShift = fillReconnectionPartialShiftRcDataPrefixPrevFrame(neighborOffset);
            }

            ReconnectionData rcDataShift;

            if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength()
                && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
            {
                rcDataShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId + 1]];
            }
            else
            {
                rcDataShift = rcDataCenter;
            }

            bool selected = StreamNeighborPathIntoPathReservoir(subpathReservoir, currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                                1.f, 
                                                                canonicalMisWeights, sg, true, doTemporalUpdate);

        };

        canonicalMisWeights = hasValidTemporalNeighbor ? canonicalMisWeights : 1.f;

        bool selected = subpathReservoir.merge(currentReservoir.integrand, 1.f, currentReservoir, sg, canonicalMisWeights);

        subpathReservoir.finalizeGRIS();
        subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                      ? 0.f
                                      : max(0.f, subpathReservoir.weight);

        subpathReservoir.pathFlags.insertPathTreeLength(currentPathTreeLength); 
        subpathReservoir.pathFlags.insertPrefixLength(currentPrefixLength);

        subpathReservoir.initRandomSeed = currentReservoir.initRandomSeed;

        reservoirs[offset] = subpathReservoir;
    }

    void runSpatial(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool centralPrefixValid = isPixelPrefixLastSdValid(offset);
        if (!centralPrefixValid) return;

        SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.suffixResamplingSgOffset() + suffixReuseRoundId);

        PathReservoir subpathReservoir;
        subpathReservoir.init();

        int vertexBufferOffset = (kMaxSurfaceBounces + 1) * offset;

        float3 color = 0.f;
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        PathReservoir currentReservoir = prevReservoirs[offset];

        int currentPathTreeLength = currentReservoir.pathFlags.pathTreeLength();
        int currentPrefixLength = currentReservoir.pathFlags.prefixLength();
        bool currentUserFlagSet = currentReservoir.pathFlags.isUserFlagSet();

        int validNeighbors = 0;

        int numNeighbors = restir.subpathSettings.suffixSpatialNeighborCount;
        int2 reprojPixel = pixel;

        float canonicalMisWeights = 1.f;

        ReconnectionData rcDataCenter = fillReconnectionPartialShiftRcDataPrefix(offset);

        for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
        {
            int2 neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, neighborId);

            if (!neighborValidMask[offset].isValid(neighborId)) continue;

            int neighborOffset = params.getReservoirOffset(neighborPixel);

            validNeighbors++;

            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            ReconnectionData rcDataBackShift;

            if (currentReservoir.pathFlags.rcVertexLength() > 1 + currentReservoir.pathFlags.prefixLength() 
                && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
            {
                rcDataBackShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId]];
            }
            else
            {
                rcDataBackShift = fillReconnectionPartialShiftRcDataPrefix(neighborOffset);
            }

            ReconnectionData rcDataShift;

            if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                && ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid)
            {
                rcDataShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId + 1]];
            }
            else
            {
                rcDataShift = rcDataCenter;
            }

            bool selected = StreamNeighborPathIntoPathReservoir(subpathReservoir, currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                                numNeighbors, 
                                                                canonicalMisWeights, sg);
        };

        bool selected = subpathReservoir.merge(currentReservoir.integrand, 1.f, currentReservoir, sg, canonicalMisWeights);

        subpathReservoir.finalizeGRIS();
        subpathReservoir.weight /= (validNeighbors + 1);
        subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                      ? 0.f
                                      : max(0.f, subpathReservoir.weight);

        subpathReservoir.pathFlags.insertPathTreeLength(currentPathTreeLength);
        subpathReservoir.pathFlags.insertPrefixLength(currentPrefixLength);
        subpathReservoir.initRandomSeed = currentReservoir.initRandomSeed;

        reservoirs[offset] = subpathReservoir;
    }
}

cbuffer CB
{
    SuffixResampling gSuffixResampling;
}

[numthreads(16, 16, 1)]
void spatial(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gSuffixResampling.runSpatial(dispatchThreadId, groupThreadIdx, groupID);
}

[numthreads(16, 16, 1)]
void temporal(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gSuffixResampling.runTemporal(dispatchThreadId, groupThreadIdx, groupID);
}

[numthreads(16, 16, 1)]
void gather(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gSuffixResampling.runGatherSuffixes(dispatchThreadId, groupThreadIdx, groupID);
}
