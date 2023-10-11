/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#include "Scene/SceneDefines.slangh"

import Rendering.Utils.PixelStats;
import Utils.Debug.PixelDebug;

import Scene.RaytracingInline;
import PathTracer;
import LoadShadingData;
import Utils.Color.ColorHelpers;
import ConditionalReSTIR;


ParameterBlock<PathTracer> gPathTracer;
Scheduler gScheduler;

struct ClosestHitQuery : PathTracer::IClosestHitQuery
{
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT, bool isPrevFrame = false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray, bool isPrevFrame=false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct Scheduler
{
    StructuredBuffer<PrefixGBuffer> prefixGbuffer;
    RWStructuredBuffer<PathReservoir> pathReservoirs;

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */

    // TODO: enable this only in an unbiased mode

    void tracePath(uint pathID, int2 pixel)
    {
        PathState path;

        // fetch from prefix G Buffer
        int offset = gPathTracer.params.getReservoirOffset(pixel);

        HitInfo prefixHit = prefixGbuffer[offset].hit;
        
        float3 wo = prefixGbuffer[offset].wo;

        if (!prefixHit.isValid()) return;

        int prefixLength = gPathTracer.restir.minimumPrefixLength;
        if (gPathTracer.restir.subpathSettings.adaptivePrefixLength)
            prefixLength = pathReservoirs[offset].pathFlags.prefixLength();


        ShadingData sd = loadShadingData(prefixHit, 0.f, -wo, false, gPathTracer.createTextureSampler(), false);
        gPathTracer.generatePathForSuffixGeneration(pathID, prefixHit, sd.computeNewRayOrigin(), path, prefixLength);


        if (prefixHit.isValid()) path.restirData.pathRcInfo.pathFlags.insertPathTreeLength(prefixLength);

        // store randomSeed for suffix
        gPathTracer.setupPathLogging(path);

        // add NEE contribution
        VisibilityQuery vq;

        gPathTracer.addNEEForPrefixLastVertex(path, sd, vq);
        

        uint suffixInitSeed = gPathTracer.getCurrentSeed(path.sg);
        // TODO: sample within specific lobe in the lobe-dependent prefix case

        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, gPathTracer.createTextureSampler());
        bool valid = gPathTracer.generateScatterRayConditionalReSTIR(sd, mi, path, false);
        
        if (!valid || !path.isActive())
        {
            path.terminate();
        }
        else
        {
            while (path.isActive())
            {
                ClosestHitQuery chq;
                gPathTracer.nextHit(path, chq, false);

                if (path.isHit())
                {
                    VisibilityQuery vq;
                    gPathTracer.handleHit(path, vq);
                }
                else
                {
                    gPathTracer.handleMiss(path);
                }
            }
        }

        // initRandomSeed doesn't matter. won't be used
        float pHat = PathRisState::toScalar(path.restirData.pathRis.integrand);
        path.restirData.pathRis.weight = pHat == 0.f ? 0.f : path.restirData.pathRis.weight / pHat;

        path.restirData.pathRis.pathFlags.insertPathTreeLength(path.restirData.pathRcInfo.pathFlags.pathTreeLength());
        path.restirData.pathRis.pathFlags.insertPrefixLength(prefixLength);
        pathReservoirs[offset].transferRisStateNoInitRandomSeed(path.restirData.pathRis, suffixInitSeed);
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        uint pathID = pixel.x | (pixel.y << 12);
        tracePath(pathID, pixel);
    }
}

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gScheduler.run(pixel);
}
