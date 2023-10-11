/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/

__exported import PathReservoir;
__exported import ConditionalReSTIR;
import Scene.ShadingData;
import LoadShadingData; 
import PathTracer;
import Utils.Sampling.TinyUniformSampleGenerator;
import Scene.RaytracingInline;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Geometry.GeometryHelpers;
import Rendering.Materials.StandardMaterialInstance;
__exported import HybridShiftTraceRays;

bool isJacobianInvalid(float Jacobian)
{
    return Jacobian <= 0.f || isnan(Jacobian) || isinf(Jacobian);
}

bool isIntegrandInvalid(float3 Integrand)
{
    return all(Integrand == 0.f) || any(isnan(Integrand) || isinf(Integrand) || Integrand < 0.f);
}

float3 shiftPath(const ConditionalReSTIR restir, const ReSTIRPathTracerParams params, inout float Jacobian,
                 inout PathReservoir srcReservoir, ReconnectionData rcData, int dstPrefixLength, uint dstPrefixComponent,
                 bool isPrevFrame = false, bool temporalUpdateForDynamicScene = false)
{
    if (!restir.kUsePrevFrameSceneData) isPrevFrame = false;

    float3 thp = rcData.pathThroughput;

    // the reconnection vertex exists
    if (rcData.rcPrevHit.isValid())
    {
        if (restir.subpathSettings.adaptivePrefixLength)
        {
            int lengthChange = dstPrefixLength - srcReservoir.pathFlags.prefixLength();

            if (srcReservoir.pathFlags.pathLength() + lengthChange > kMaxSurfaceBounces)
                return 0.f;

            srcReservoir.pathFlags.insertPrefixLength(dstPrefixLength);
            srcReservoir.pathFlags.insertRcVertexLength(srcReservoir.pathFlags.rcVertexLength() + lengthChange);
            srcReservoir.pathFlags.insertPathLength(srcReservoir.pathFlags.pathLength() + lengthChange);
        }

        bool isRcVertexLengthOne = srcReservoir.pathFlags.rcVertexLength() == 1;

        float reconnectionJacobian = 1.f;

        ShadingData dstRcPrevSd = loadShadingData(rcData.rcPrevHit, float3(0.f), -rcData.rcPrevWo, isRcVertexLengthOne, ExplicitLodTextureSampler(isRcVertexLengthOne ? 0 : params.lodBias), isPrevFrame);
        if (temporalUpdateForDynamicScene && restir.kTemporalUpdateForDynamicScene)
        {
            TraceTemporalUpdate(dstRcPrevSd, srcReservoir, isPrevFrame);
            if (all(srcReservoir.rcIrrad == 0.f)) return 0.f;
        }
        thp *= shiftPathReconnection(restir, params, reconnectionJacobian, dstRcPrevSd, srcReservoir, isPrevFrame);
        
        Jacobian *= reconnectionJacobian;
    }

    return thp;
}

float3 shiftPathReconnection(const ConditionalReSTIR restir, const ReSTIRPathTracerParams params, inout float Jacobian, ShadingData dstPrimarySd,
                             inout PathReservoir srcReservoir, bool isPrevFrame) // for updating vertex buffer
{
    var sg = TinyUniformSampleGenerator(1);

    const int rcVertexLength = srcReservoir.pathFlags.rcVertexLength();
    const int prefixLength = srcReservoir.pathFlags.prefixLength();
    
    const HitInfo rcVertexHit = srcReservoir.rcHit;
    
    const float3 rcVertexIrradiance = srcReservoir.rcIrrad;
    const float3 rcVertexWi = srcReservoir.rcWi;
    const bool rcVertexHitExists = rcVertexHit.isValid();
    const float3 rcJacobian = srcReservoir.rcJacobian;

    const bool isTransmission1 = srcReservoir.pathFlags.decodeIsTransmissionEvent(true);
    const uint componentType1 = srcReservoir.pathFlags.decodeBSDFComponent(true);

    const bool isDelta1 = srcReservoir.pathFlags.decodeIsDeltaEvent(true);
    const bool isDelta2 = srcReservoir.pathFlags.decodeIsDeltaEvent(false);
    // delta bounce before/after rcVertex
    if (isDelta1 || isDelta2) return 0.f;

    const bool isRcVertexConnectedtoLight = srcReservoir.pathFlags.pathLength() == rcVertexLength;
    const bool isRcVertexEmissive = srcReservoir.pathFlags.pathLength() + 1 == rcVertexLength;
    const bool isLightPathNEE = srcReservoir.pathFlags.lastVertexNEE();
    const bool isRcVertexNEE = isRcVertexConnectedtoLight && isLightPathNEE;

    // a simple way to set ray offset. Note that this will disallow SpecularReflectionTransmission to switch side (reflection<->refraction)
    // during reuse since self-shadowing will happen
    const IMaterialInstance dstMI = gScene.materials.getMaterialInstance(dstPrimarySd, ExplicitLodTextureSampler(rcVertexLength == 1 ? 0.f : params.lodBias));
    dstPrimarySd.posW = dstPrimarySd.computeNewRayOrigin(!isTransmission1);
    float3 dstPrimaryPosW = dstPrimarySd.posW;

    ShadingData rcVertexSd = {};
    float3 shiftedDisp = 0.f;
    float shifted_dist2 = 0.f;

    // direction point from dst primary hit point to rcVertex
    float3 dstConnectionV;

    rcVertexSd = loadShadingData(rcVertexHit, dstPrimaryPosW, float3(0.f), false, ExplicitLodTextureSampler(params.lodBias), isPrevFrame);
    const IMaterialInstance rcMI = gScene.materials.getMaterialInstance(rcVertexSd, ExplicitLodTextureSampler(params.lodBias));
    float3 rcVertexPosW = rcVertexSd.posW;
    float3 rcFaceN = rcVertexSd.faceN;

    shiftedDisp = rcVertexPosW - dstPrimaryPosW;
    shifted_dist2 = dot(shiftedDisp, shiftedDisp);

    // check if the shift violates roughness/distance condition
    bool violateRoughnessCondition = rcVertexLength == prefixLength + 1 && !(componentType1 >= 2 ? dstMI.getProperties(dstPrimarySd).roughness > restir.settings.specularRoughnessThreshold : 
                             (dstMI.getLobeTypes(dstPrimarySd) & (uint)LobeType::Diffuse) != 0);

    bool violateDistanceCondition = rcVertexHitExists && shifted_dist2 <= restir.settings.nearFieldDistanceThreshold * restir.sceneRadius * restir.settings.nearFieldDistanceThreshold * restir.sceneRadius;

    Jacobian *= ConditionalReSTIR::ShiftMapping(restir.shiftMapping) == ConditionalReSTIR::ShiftMapping::Hybrid && (violateDistanceCondition || violateRoughnessCondition) && !(isRcVertexEmissive && isLightPathNEE) ? 0.f : 1.f;

    dstConnectionV = rcVertexHitExists ? shiftedDisp / sqrt(shifted_dist2) : rcVertexWi;

    float shifted_cosine = abs(dot(rcFaceN, -dstConnectionV));
    float dstJacobian = rcVertexHitExists ? (shifted_cosine / shifted_dist2) : 1.f;

    float3 dstIntegrand = rcVertexIrradiance;

    float dstPDF1BSDF, dstPDF1ForMIS;

    float3 dstF1 = dstMI.evalBsdfAndPdf(dstPrimarySd, dstConnectionV, sg, isRcVertexEmissive && isLightPathNEE ? 0xf : 1 << componentType1, dstPDF1BSDF, dstPDF1ForMIS);
    bool rcFrontFacing = rcVertexSd.frontFacing;

    // If we use a position-dependent NEE sampler, then NEE pdf needs to be re-evaluated
    // lightPdf stored in area measure if rcVertexHitExists
    float lightPDF1 = srcReservoir.lightPdf / dstJacobian;
    float dstPDF1 = isRcVertexEmissive && isLightPathNEE ? lightPDF1 : dstPDF1BSDF;

    dstJacobian *= dstPDF1;
    dstIntegrand *= dstF1 / dstPDF1;

    dstIntegrand *= uint(!rcVertexHitExists || !isRcVertexEmissive || rcFrontFacing);

    float MISForEmissive = !kUseMIS ? 1.f : PathTracer::evalMIS(1, isLightPathNEE ? lightPDF1 : dstPDF1ForMIS, 1, isLightPathNEE ? dstPDF1ForMIS : lightPDF1);    
    dstIntegrand *= isRcVertexEmissive ? MISForEmissive : 1.f;

    const bool needSecondPart = rcVertexHitExists && !isRcVertexEmissive;

    const uint componentType2 = srcReservoir.pathFlags.decodeBSDFComponent(false);
    float dstPDF2BSDF, dstPDF2ForMIS;

    float3 dstF2 = rcMI.evalBsdfAndPdf(rcVertexSd, rcVertexWi, sg, isRcVertexNEE ? 0xf : 1 << componentType2, dstPDF2BSDF, dstPDF2ForMIS);

    float MISForLightVertex = (kUseMIS && isRcVertexConnectedtoLight && srcReservoir.pathFlags.lightType() != (uint)PathTracer::LightType::Analytic) ? 
    PathTracer::evalMIS(1, isRcVertexNEE ? srcReservoir.lightPdf : dstPDF2ForMIS, 1, isRcVertexNEE ? dstPDF2ForMIS : srcReservoir.lightPdf) : 1;

    float dstPDF2 = isRcVertexNEE ? srcReservoir.lightPdf : dstPDF2BSDF;
    dstJacobian *= !needSecondPart ? 1.f : dstPDF2;

    dstIntegrand *= MISForLightVertex;
    dstIntegrand *= needSecondPart ? (dstF2 / dstPDF2) : 1.f;

    Jacobian *= dstJacobian / srcReservoir.rcJacobian;

    float3 shadowDestination = rcVertexHitExists ? rcVertexPosW : rcVertexWi;
    bool isVisible = evalSegmentVisibility(isPrevFrame, dstPrimaryPosW, shadowDestination, !rcVertexHitExists);
    bool isInvalidJacobianOrIntegrand = isJacobianInvalid(Jacobian) || isIntegrandInvalid(dstIntegrand);
    bool invalidShift = !isVisible || isInvalidJacobianOrIntegrand;

    Jacobian = invalidShift ? 0.f : Jacobian;
    dstIntegrand = invalidShift ? 0.f : dstIntegrand;

    srcReservoir.rcJacobian = dstJacobian;

    return dstIntegrand;
}

/// Evaluates the visibility term to the reconnection vertex from the source vertex.
bool evalSegmentVisibility(bool isPrevFrame, const float3 sourceVertex, const float3 targetVertex, bool isDirection = false)
{
    float3 dir = isDirection ? targetVertex : targetVertex - sourceVertex;
    float tMax = isDirection ? kRayTMax : 0.999f;

    Ray ray = { sourceVertex, 0.f, dir, tMax }; // TODO: use offset position instead

    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    logTraceRay(PixelStatsRayType::Visibility);    
    return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
}
