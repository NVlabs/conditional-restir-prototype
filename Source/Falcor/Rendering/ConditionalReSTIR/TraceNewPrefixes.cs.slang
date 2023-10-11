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
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT, bool isPrevFrame=false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray, bool isPrevFrame = false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct Scheduler
{
    //StructuredBuffer<PrefixGBuffer> prefixGbuffer;
    //RWStructuredBuffer<PathReservoir> pathReservoirs;

    int integrationPrefixId;
    bool shouldGenerateSuffix;

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */

    // TODO: enable this only in an unbiased mode

    void tracePath(uint pathID)
    {
        PathState path;
        gPathTracer.generatePath(pathID, path, shouldGenerateSuffix);
        gPathTracer.setupPathLogging(path);

        // Note the primary miss has already been handled by the separate path generation pass
        // the path tracer runs first. Abort execution here to avoid double work.
        if (!path.isHit()) return;

        while (path.isActive())
        {
            if (path.isHit())
            {
                VisibilityQuery vq;
                gPathTracer.handleHit(path, vq);
                if (path.isTerminated()) break;

                ClosestHitQuery chq;
                gPathTracer.nextHit(path, chq, false);
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

        gPathTracer.writeSuffixReservoir(path);
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        uint pathID = pixel.x | (pixel.y << 12) | (integrationPrefixId << 24);
        tracePath(pathID);
    }
}

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gScheduler.run(pixel);
}
