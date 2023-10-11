/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/

#include "Utils/Math/MathConstants.slangh"

import PathReservoir;
import Utils.Sampling.SampleGenerator;
import LoadShadingData;
import Params;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Scene.RaytracingInline;
import Scene.Scene;
import PathTracer;
import RenderPasses.Shared.Denoising.NRDData;
import Rendering.Lights.EmissiveLightSampler;
import RenderPasses.Shared.Denoising.NRDConstants;
import Rendering.Materials.IsotropicGGX;

ParameterBlock<PathTracer> gPathTracer;

struct ClosestHitQuery : PathTracer::IClosestHitQuery
{
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT, bool isPrevFrame)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray, bool isPrevFrame)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

float3 TraceReplayedPath(

    HitInfo dstPrimaryHit, ShadingData dstPrimarySd,
    
    ReSTIRPathFlags pathFlags, uint initRandomSeed, uint suffixInitRandomSeed, 
                         
        inout HitInfo rcPrevHit, 
        
        inout float3 rcPrevHitWo, bool isPrevFrame, out float prefixPartTotalLength, bool isSuffixReplay=false, bool isPrefixReplay=false)
{
    if (pathFlags.pathLength() == 0) return 0.f;
    if (!gPathTracer.restir.kUsePrevFrameSceneData) isPrevFrame = false;

    PathState path;

    gPathTracer.generatePathForRandomReplay(dstPrimaryHit, dstPrimarySd, path, initRandomSeed, isSuffixReplay, suffixInitRandomSeed, pathFlags);
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(dstPrimarySd, ExplicitLodTextureSampler(isSuffixReplay ? gPathTracer.params.lodBias : 0));
    bool valid = gPathTracer.generateScatterRayConditionalReSTIR(dstPrimarySd, mi, path, true);

    path.setPrevFrame(isPrevFrame);

    path.setPrefixReplay(isPrefixReplay);

    path.pathTotalLength = 0.f;

    if (!valid || !path.isActive())
    {
        return 0.f;
    }

    bool terminateWithMiss = false;

    while (path.isActive())
    {
        ClosestHitQuery chq;
        gPathTracer.nextHit(path, chq, isPrevFrame);

        if (path.isHit())
        {
            VisibilityQuery vq;
            gPathTracer.handleHitReplayPath(path, vq);
        }
        else
        {
            terminateWithMiss = true;
            gPathTracer.handleMissReplayPath(path);
        }
    }

    bool shouldConnect = pathFlags.rcVertexLength() <= pathFlags.pathLength() + 1;
    bool readyToConnect = path.getVertexIndex() - 1 == pathFlags.rcVertexLength() - 1;

    // Check if we need to terminate due to non-invertible shift
    bool nonInvertable = (shouldConnect && (!readyToConnect || terminateWithMiss)) || path.getVertexIndex() == 0;

    float3 L = nonInvertable ? 0.f : path.L;

    if (shouldConnect && !nonInvertable)
    {
        rcPrevHit = path.pathReplayInfo.rcHit;
        rcPrevHitWo = path.pathReplayInfo.rcWi;
        L = path.thp;
    }

    prefixPartTotalLength = path.pathTotalLength;

    return L;
};


void TraceTemporalUpdate(ShadingData rcPrevSd, inout PathReservoir reservoir, bool isPrevFrame)
{
    if (!gPathTracer.restir.kUsePrevFrameSceneData) isPrevFrame = false;

    int rcVertexLength = reservoir.pathFlags.rcVertexLength();
    if (!reservoir.rcHit.isValid()) // rcVertex is environment light
    {
        assert(reservoir.pathFlags.pathLength() + 1 == rcVertexLength);
        reservoir.rcIrrad = gPathTracer.envMapSampler.eval(reservoir.rcWi);
        reservoir.lightPdf = gPathTracer.getEnvMapSelectionProbability() * gPathTracer.envMapSampler.evalPdf(reservoir.rcWi);
        return;
    }

    ShadingData rcVertexSd = loadShadingData(reservoir.rcHit, rcPrevSd.posW, float3(0.f), false, ExplicitLodTextureSampler(gPathTracer.params.lodBias), isPrevFrame);
    
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(rcVertexSd, ExplicitLodTextureSampler(gPathTracer.params.lodBias));

    uint rcRandomSeed = 0;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
    rcRandomSeed = reservoir.rcRandomSeed;
#endif

    PathState path;

    gPathTracer.generatePathForTemporalUpdate(reservoir.rcHit, rcVertexSd, path, reservoir.rcWi, rcRandomSeed, reservoir.pathFlags);
    
    path.setPrevFrame(isPrevFrame);

    if (reservoir.pathFlags.lastVertexNEE() && reservoir.pathFlags.pathLength() == rcVertexLength)
    {
        VisibilityQuery vq;
        gPathTracer.updateDirectLightingForRcVertex(path, rcVertexSd, mi, vq, true, isPrevFrame);
        reservoir.rcIrrad = path.pathReplayInfo.rcThroughput;
        reservoir.lightPdf = path.pathReplayInfo.rcJacobian;
        reservoir.rcWi = path.dir;
        reservoir.pathFlags.insertLightType(path.pathReplayInfo.pathFlags.lightType());
    }
    else if (reservoir.pathFlags.pathLength() + 1 != rcVertexLength)
    {
        int rcVertexLength = reservoir.pathFlags.rcVertexLength();
        bool valid = gPathTracer.generateScatterRayGivenDirection(rcVertexSd, mi, path);
        path.pathReplayInfo.rcThroughput = float3(1.f); // reset throughput

        // trace stuff to update stuff
        if (!valid) path.terminate();

        while (path.isActive())
        {
            ClosestHitQuery chq;
            gPathTracer.nextHit(path, chq, isPrevFrame);

            if (path.isHit())
            {
                VisibilityQuery vq;
                gPathTracer.handleHitReplayPath(path, vq);
            }
            else
            {
                gPathTracer.handleMissReplayPath(path);
            }
        }

        // Now we don't have NEE in handleHitReplayPath, so we have to do it here
        if (reservoir.pathFlags.lastVertexNEE())
        {
            ShadingData NEESd = loadShadingData(path.hit, path.origin, path.dir, false, ExplicitLodTextureSampler(gPathTracer.params.lodBias), isPrevFrame);
            const IMaterialInstance NEEmi = gScene.materials.getMaterialInstance(NEESd, ExplicitLodTextureSampler(gPathTracer.params.lodBias));
            VisibilityQuery vq;
            gPathTracer.updateDirectLightingForRcVertex(path, NEESd, NEEmi, vq, false, isPrevFrame);
        }

        if (any(path.pathReplayInfo.rcThroughput > 0.f) && path.hasNonZeroTemporalUpdate())
            reservoir.rcIrrad = path.pathReplayInfo.rcThroughput;
        reservoir.rcIrrad = path.hasNonZeroTemporalUpdate() ? path.pathReplayInfo.rcThroughput : 0.f;
        reservoir.lightPdf = path.pathReplayInfo.rcJacobian;
        reservoir.pathFlags.insertLightType(path.pathReplayInfo.pathFlags.lightType());
    }

    // WARNING: assuming envmap doesn't rotate
    // WARNING: assuming emission doesn't change
    // WARNING: assuming light PDF doesn't change with frame!
}
