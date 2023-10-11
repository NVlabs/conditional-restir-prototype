/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#include "Scene/SceneDefines.slangh"

import Rendering.Utils.PixelStats;
import PathTracerNRD;

import Scene.RaytracingInline;
import PathTracer;

ParameterBlock<PathTracer> gPathTracer;
Scheduler gScheduler;

struct ClosestHitQuery : PathTracer::IClosestHitQuery
{
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, false);
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, false);
    }
};

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct Scheduler
{
    uint curIter;
    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path;
        gPathTracer.generatePath(pathID, path);

        // don't want log ray stats for denoiser related stuff
#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.setupPathLogging(path);
#endif

        // Note the primary miss has already been handled by the separate path generation pass
        // the path tracer runs first. Abort execution here to avoid double work.
        if (!path.isHit()) return;

        while (path.isActive())
        {
            if (path.isHit())
            {
                VisibilityQuery vq;
#if defined(DELTA_REFLECTION_PASS)
                gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
                gPathTracer.handleDeltaTransmissionHit(path);
#else
                gPathTracer.handleHit(path, vq);
#endif
                if (path.isTerminated()) break;

                ClosestHitQuery chq;
                gPathTracer.nextHit(path, chq);
            }
            else
            {
                gPathTracer.handleMiss(path);
            }
            // write to the prefix reservoir 
        }

        // write to prefix reservoir
        if (!path.restirData.pathRcInfo.pathFlags.isUserFlagSet()) // haven't been written 
        {
            gPathTracer.writePrefixPathReservoir(path);
        }

        // write to the suffix reservoir

#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.writeOutput(path);
#endif
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
#if defined(DELTA_REFLECTION_PASS) || defined(DELTA_TRANSMISSION_PASS)
        uint pathID = pixel.x | (pixel.y << 12);
        tracePath(pathID);
#else
        if (gPathTracer.params.samplesPerPixel == 1)
        {
            // Handle fixed 1 spp case.
            uint pathID = pixel.x | (pixel.y << 12);
            tracePath(pathID);
        }
        else
        {
            uint samples = gPathTracer.params.samplesPerPixel;
            if (gPathTracer.params.samplesPerPixel == 0)
                samples = gPathTracer.sampleCount[pixel];
            if (gPathTracer.params.useConditionalReSTIR) samples = 1;

            for (uint sampleIdx = 0; sampleIdx < samples; sampleIdx++)
            {
                uint pathID = pixel.x | (pixel.y << 12) | ((gPathTracer.params.useConditionalReSTIR ? curIter : sampleIdx) << 24);
                tracePath(pathID);
            }
        }
#endif
    }
}

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gScheduler.run(pixel);
}
