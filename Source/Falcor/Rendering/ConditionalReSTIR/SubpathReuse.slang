
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
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import ConditionalReSTIR;
import Shift;

static const uint kGroupLen = 16;
static const uint kGroupSize = kGroupLen * kGroupLen;

float computeCanonicalPairwiseMISWeight(float3 basisPathContributionAtBasis, float3 basisPathContributionAtNeighbor, float basisPathToNeighborJacobian,
                                        float pairwiseK, float canonicalM, float neighborM)
{
    float misWeightBasisPath = 1.f;

    if (PathRisState::toScalar(basisPathContributionAtBasis) > 0.f)
    {
        float atBasisTerm = PathRisState::toScalar(basisPathContributionAtBasis) * canonicalM;
        misWeightBasisPath = atBasisTerm / (atBasisTerm + PathRisState::toScalar(basisPathContributionAtNeighbor) * basisPathToNeighborJacobian * neighborM * pairwiseK);
    }
    return misWeightBasisPath;
}

float computeNeighborPairwiseMISWeight(float3 neighborPathContributionAtBasis, float3 neighborPathContributionAtNeighbor,
                                       float neighborPathToBasisJacobian, float pairwiseK, float canonicalM, float neighborM)
{
    float misWeightNeighborPath = 0.f;
    if (PathRisState::toScalar(neighborPathContributionAtNeighbor) > 0.f)
    {
        misWeightNeighborPath = PathRisState::toScalar(neighborPathContributionAtNeighbor) * neighborM /
                                (PathRisState::toScalar(neighborPathContributionAtNeighbor) * neighborM + PathRisState::toScalar(neighborPathContributionAtBasis) * neighborPathToBasisJacobian * canonicalM / pairwiseK);
    }
    return misWeightNeighborPath;
}

float GeometryTerm(float3 x0, float3 x1, float3 x1faceN)
{
    float3 disp = x0 - x1;
    float d2 = dot(disp, disp);
    float3 dir = disp / sqrt(d2);
    return abs(dot(dir, x1faceN)) / d2;
}

uint decodeLobeType(inout BSDFSample bs)
{
    uint componentType = 0;
    if (bs.isLobe(LobeType::DiffuseReflection) && (bs.isLobe(LobeType::Specular) || bs.isLobe(LobeType::Delta)))
    { // we use this to differentiate SpecularReflectionTransmission from SpecularReflection
        // for Standard Material Only
        componentType = 3;
        bs.lobeType &= ~((uint)LobeType::DiffuseReflection);
    }
    else if (!bs.isLobe(LobeType::DiffuseReflection))
    {
        componentType = bs.isLobe(LobeType::DiffuseTransmission)
                            ? 1 : componentType;
        componentType = (bs.isLobe(LobeType::Specular) || bs.isLobe(LobeType::Delta))
                            ? 2 : componentType;
    }

    return componentType;
}

float computeTalbotMISWeightTerm(PathReservoir neighborReservoir,
                                PathReservoir dstReservoir,
                                ReconnectionData rcDataShift,
                                bool doPrint = false)
{
    float jacobianShift = 1.f;
    float3 integrandShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianShift, neighborReservoir,
        rcDataShift, dstReservoir.pathFlags.prefixLength(), 0);
    return PathRisState::toScalar(integrandShift) * jacobianShift * dstReservoir.M;
}

float3 integrateWithNeighborReservoirSampleTalbotMIS(
                                                  PathReservoir centralReservoir,
                                                  PathReservoir neighborReservoir,
                                                  ReconnectionData rcDataShift,
                                                  float talbotMISWeightPartialSum)
{
    float jacobianShift = 1.f;
    int neighborReservoirPrefixLength = neighborReservoir.pathFlags.prefixLength(); // neighborReservoir is overwritten, so we have to save this
    uint neighborReservoirPrefixComponent = neighborReservoir.pathFlags.isUserFlagSet() ? 4 : neighborReservoir.pathFlags.decodeBSDFComponent(true);
    float3 integrandShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianShift, neighborReservoir,
                                      rcDataShift, centralReservoir.pathFlags.prefixLength(), centralReservoir.pathFlags.isUserFlagSet() ? 4 : centralReservoir.pathFlags.decodeBSDFComponent(true));

    print("neighbor integrandShift", integrandShift);
    print("neighbor original", neighborReservoir.integrand);

    float selfWeight = PathRisState::toScalar(neighborReservoir.integrand) * neighborReservoir.M;
    float neighborMisWeight = selfWeight 
                            / (talbotMISWeightPartialSum + selfWeight + PathRisState::toScalar(integrandShift) * jacobianShift * centralReservoir.M);

    float3 contrib = integrandShift * jacobianShift * neighborReservoir.weight * neighborMisWeight;
    contrib = any(isnan(contrib) || isinf(contrib)) ? 0.0f : contrib;

    return contrib;
}

bool StreamNeighborPathIntoPathReservoirTalbotMIS(inout PathReservoir reservoir,
                                                    PathReservoir centralReservoir,
                                                    PathReservoir neighborReservoir,
                                                    ReconnectionData rcDataShift,
                                                    float talbotMISWeightPartialSum,
                                                    inout SampleGenerator sg,
                                                    bool doPrint = false)
{
    float jacobianShift = 1.f;
    int neighborReservoirPrefixLength = neighborReservoir.pathFlags.prefixLength(); // neighborReservoir is overwritten, so we have to save this
    uint neighborReservoirPrefixComponent = neighborReservoir.pathFlags.isUserFlagSet() ? 4 : neighborReservoir.pathFlags.decodeBSDFComponent(true);
    float3 integrandShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianShift, neighborReservoir,
            rcDataShift, centralReservoir.pathFlags.prefixLength(), centralReservoir.pathFlags.isUserFlagSet() ? 4 : centralReservoir.pathFlags.decodeBSDFComponent(true));

    print("neighbor integrandShift", integrandShift);
    print("neighbor original", neighborReservoir.integrand);

    float selfWeight = PathRisState::toScalar(neighborReservoir.integrand) * neighborReservoir.M;
    float neighborMisWeight = selfWeight 
                            / (talbotMISWeightPartialSum + selfWeight + PathRisState::toScalar(integrandShift) * jacobianShift * centralReservoir.M);

    bool selected = reservoir.merge(integrandShift, jacobianShift, neighborReservoir, sg, neighborMisWeight);
    return selected;
}

float3 integrateWithNeighborReservoirSample(PathReservoir centralReservoir,
                                          PathReservoir neighborReservoir,
                                          ReconnectionData rcDataShift,
                                          ReconnectionData rcDataBackShift,
                                          float pairwiseK,
                                          inout float canonicalMisWeights)
{
    float jacobianShift = 1.f;
    float jacobianBackShift = 1.f;
    int neighborReservoirPrefixLength = neighborReservoir.pathFlags.prefixLength(); // neighborReservoir is overwritten, so we have to save this
    uint neighborReservoirPrefixComponent = neighborReservoir.pathFlags.isUserFlagSet() ? 4 : neighborReservoir.pathFlags.decodeBSDFComponent(true);
    float3 integrandShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianShift, neighborReservoir,
                                      rcDataShift, centralReservoir.pathFlags.prefixLength(), centralReservoir.pathFlags.isUserFlagSet() ? 4 : centralReservoir.pathFlags.decodeBSDFComponent(true));
    float3 integrandBackShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianBackShift, centralReservoir, rcDataBackShift, neighborReservoirPrefixLength, neighborReservoirPrefixComponent);

    float neighborMisWeight = computeNeighborPairwiseMISWeight(integrandShift, neighborReservoir.integrand, jacobianShift, pairwiseK, centralReservoir.M, neighborReservoir.M);

    float canonicalMisWeight = computeCanonicalPairwiseMISWeight(centralReservoir.integrand, integrandBackShift, jacobianBackShift, pairwiseK, centralReservoir.M, neighborReservoir.M);

    canonicalMisWeights += canonicalMisWeight;
    // don't update central's initRandomSeed

    float3 contrib = integrandShift * jacobianShift * neighborReservoir.weight * neighborMisWeight;
    contrib = any(isnan(contrib) || isinf(contrib)) ? 0.0f : contrib;

    return contrib;
}

bool StreamNeighborPathIntoPathReservoir(inout PathReservoir reservoir,
                                         PathReservoir centralReservoir,
                                         PathReservoir neighborReservoir,
                                         ReconnectionData rcDataShift,
                                         ReconnectionData rcDataBackShift,
                                         float pairwiseK,
                                         inout float canonicalMisWeights,
                                         inout SampleGenerator sg,
                                         bool isBackShiftToPrevFrame = false,
                                         bool temporalUpdateForDynamicScene = false)
{
    //
    float jacobianShift = 1.f;
    float jacobianBackShift = 1.f;
    int neighborReservoirPrefixLength = neighborReservoir.pathFlags.prefixLength(); // neighborReservoir is overwritten, so we have to save this
    uint neighborReservoirPrefixComponent = neighborReservoir.pathFlags.isUserFlagSet() ? 4 : neighborReservoir.pathFlags.decodeBSDFComponent(true);
    float3 integrandShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianShift, neighborReservoir,
            rcDataShift, centralReservoir.pathFlags.prefixLength(), centralReservoir.pathFlags.isUserFlagSet() ? 4 : centralReservoir.pathFlags.decodeBSDFComponent(true), false, temporalUpdateForDynamicScene);
    float3 integrandBackShift = shiftPath(gPathTracer.restir, gPathTracer.params, jacobianBackShift, centralReservoir, rcDataBackShift,
                                          neighborReservoirPrefixLength, neighborReservoirPrefixComponent, isBackShiftToPrevFrame, temporalUpdateForDynamicScene);

    float neighborMisWeight = computeNeighborPairwiseMISWeight(integrandShift, neighborReservoir.integrand, jacobianShift, pairwiseK, centralReservoir.M, neighborReservoir.M);

    float canonicalMisWeight = computeCanonicalPairwiseMISWeight(centralReservoir.integrand, integrandBackShift, jacobianBackShift, pairwiseK, centralReservoir.M, neighborReservoir.M);

    canonicalMisWeights += canonicalMisWeight;

    bool selected = reservoir.merge(integrandShift, jacobianShift, neighborReservoir, sg, neighborMisWeight);
    return selected;
}



float3 shiftPrefixReconnection(
    ShadingData currentPrimarySd, IMaterialInstance currentMi,
    VertexData prefixVd, inout LastVertexState lastVertexState, inout SampleGenerator sg,
    inout float jacobian, inout float srcJacobian, inout float3 newWo, inout float newPdf, inout float pathFootprint, bool shiftToPrevFrame)
{
    // using bsdf as target function
    //
    float3 connectDirCurrent = normalize(prefixVd.posW - currentPrimarySd.posW);

    int componentType = lastVertexState.lastBSDFComponent();

    float pdfCurrent, pdfAllLobesCurrent;
    float3 fCurrent = currentMi.evalBsdfAndPdf(currentPrimarySd, connectDirCurrent, sg, 1 << componentType, pdfCurrent, pdfAllLobesCurrent);
    newPdf = pdfCurrent;

    bool isRoughBounce = true;
    bool isDistant = true;
    float roughness = currentMi.getProperties(currentPrimarySd).roughness;

    if (componentType >= 2 && gPathTracer.restir.subpathSettings.avoidSpecularPrefixEndVertex)
        isRoughBounce = roughness > gPathTracer.restir.settings.specularRoughnessThreshold;

    if (gPathTracer.restir.subpathSettings.avoidShortPrefixEndSegment)
        isDistant = length(prefixVd.posW - currentPrimarySd.posW) > gPathTracer.restir.subpathSettings.shortSegmentThreshold * gPathTracer.restir.sceneRadius;

    if (!isRoughBounce || !isDistant) return 0.f;

    if (pdfCurrent > 0.f) fCurrent /= pdfCurrent;

    float geometryCurrent = GeometryTerm(currentPrimarySd.posW, prefixVd.posW, prefixVd.faceNormalW);

    jacobian = geometryCurrent * pdfCurrent / srcJacobian;
    srcJacobian = geometryCurrent * pdfCurrent;
    if (isnan(jacobian) || isinf(jacobian)) jacobian = 0.f;

    newWo = -connectDirCurrent;
    // test visibility
    bool visible = evalSegmentVisibility(shiftToPrevFrame, currentPrimarySd.computeNewRayOrigin(prefixVd.posW), prefixVd.posW);

    float3 currentToPrefixDir = prefixVd.posW - currentPrimarySd.posW;

    pathFootprint += length(currentToPrefixDir);

    return visible ? fCurrent : 0.f;
}

// prev frame data not supported is shiftPrefixReplay
float3 shiftPrefixReplay(

    ShadingData currentPrimarySd, IMaterialInstance currentMi,

    inout SampleGenerator sg,
    inout LastVertexState lastVertexState, 
    inout HitInfo prefixHit,
    inout float newPdf,
    inout float3 newWo, inout float pathFootprint)
{
    BSDFSample bs = {};

    currentPrimarySd.mtl.setLobeMask(1);
    bool isBSDFSampleValid = currentMi.sample(currentPrimarySd, sg, bs);

    if (!isBSDFSampleValid) return 0.f;

    newPdf = bs.pdf;
    newWo = -bs.wo;

    // get total pdf in a separate call
    uint newComponentType = bs.getComponentType();
    // ReSTIR
    bool isRoughBounce = bs.isLobe(LobeType::Diffuse) || bs.isLobe(LobeType::Specular) &&

        currentMi.getProperties(currentPrimarySd).roughness > gPathTracer.restir.settings.specularRoughnessThreshold;
    Ray scatterRay = Ray(currentPrimarySd.computeNewRayOrigin(!bs.isLobe(LobeType::Transmission)), bs.wo, 0.f, kRayTMax);
                                                             
    ClosestHitQuery chq;
    //
    float hitT;
    PathState dummy = {};
    chq.traceRay(dummy, scatterRay, prefixHit, hitT, false);

    if (prefixHit.isValid())
    {
        VertexData prefixHitVd = loadVertexData(prefixHit);

        bool isDistant = length(prefixHitVd.posW - currentPrimarySd.posW) > gPathTracer.restir.settings.nearFieldDistanceThreshold * gPathTracer.restir.sceneRadius;
        float3 toPrefixHitDir = prefixHitVd.posW - currentPrimarySd.posW;

        pathFootprint += length(toPrefixHitDir);

        if (isRoughBounce && isDistant) prefixHit = {}; //invalid!

        const uint lobeTypes = currentMi.getLobeTypes(currentPrimarySd);
        
        bool isNEEVertexClassifiedAsRough = (lobeTypes & (uint)LobeType::Diffuse) != 0 || isRoughBounce;
        lastVertexState.Init(isDistant, newComponentType, bs.isLobe(LobeType::Delta), bs.isLobe(LobeType::Transmission), isRoughBounce, isNEEVertexClassifiedAsRough);
    }

    return prefixHit.isValid() ? bs.weight : 0.f;
}


float3 shiftPrefix(ShadingData currentPrimarySd, HitInfo currentPrimaryHit, uint neighborInitRandomSeed, uint prefixLength, HitInfo neighborPrefixHit,
                   SampleGenerator sg, inout LastVertexState lastVertexState, bool needRandomReplay, inout float jacobian, inout float srcJacobian, inout float3 newWo,
                   inout HitInfo prefixHit, inout float newPdf, inout float pathFootprint, bool shiftToPrevFrame)
                   
{

    if (!gPathTracer.restir.kUsePrevFrameSceneData) shiftToPrevFrame = false;

    newWo = 0.f; prefixHit = {}; jacobian = 0.f; newPdf = 0.f;
    
    sg = SampleGenerator(neighborInitRandomSeed);
    const ExplicitLodTextureSampler texSampler = ExplicitLodTextureSampler(0);

    IMaterialInstance currentMi = gScene.materials.getMaterialInstance(currentPrimarySd, texSampler);
    float3 x0x1 = currentPrimarySd.posW - gScene.camera.data.posW;

    bool shouldReconnect = true;

    if (!gPathTracer.restir.subpathSettings.adaptivePrefixLength)
    {
        shouldReconnect = !needRandomReplay;
    }
    else if (prefixLength > 1)
    {
        shouldReconnect = lastVertexState.lastBSDFComponent() >= 2 ? currentMi.getProperties(currentPrimarySd).roughness > gPathTracer.restir.settings.specularRoughnessThreshold : 
                            (currentMi.getLobeTypes(currentPrimarySd) & (uint)LobeType::Diffuse) != 0;
    }

    if (!neighborPrefixHit.isValid())
    {
        return 0.f;
    }

    float3 ret = 0.f;

    if (shouldReconnect)
    {
        prefixHit = neighborPrefixHit;

        VertexData neighborPrefixVd = loadVertexData(neighborPrefixHit, shiftToPrevFrame);
        ret = shiftPrefixReconnection(currentPrimarySd, currentMi, neighborPrefixVd, lastVertexState, sg,
                                      jacobian, srcJacobian, newWo, newPdf, pathFootprint, shiftToPrevFrame);
    }
    else
    {
        // invalid prefix shift
        if (gPathTracer.restir.subpathSettings.adaptivePrefixLength) return 0.f;
        jacobian = 1.f;
        ret = shiftPrefixReplay(currentPrimarySd, currentMi, sg, lastVertexState, prefixHit, newPdf, newWo, pathFootprint);
    }

    return ret;
}

bool ResamplePrefix(
                    bool isNeighborValid,
                    inout PathReservoir reservoir,
                    PathReservoir neighborReservoir,
                    int temporalHistoryLength,

                    HitInfo currentPrimaryHit,
                    ShadingData currentPrimarySd,
                    HitInfo neighborPrimaryHit,
                    ShadingData neighborPrimarySd,                    

                    float3 currentReplayThp,
                    float3 neighborReplayThp,
                    inout PrefixGBuffer currentPrefix,
                    inout PrefixReservoir currentPrefixReservoir, 
                    PrefixGBuffer neighborPrefix, 
                    PrefixReservoir neighborPrefixReservoir,
                    inout SampleGenerator sg, int scratchVertexBufferOffset, inout float pathFootprint)
{
    float jacobianShift = 1.f;
    float3 woShifted;

    HitInfo shiftedPrefixHit;
    
    float pdfShift;
    neighborPrefixReservoir.setM(min(temporalHistoryLength * currentPrefixReservoir.M(), neighborPrefixReservoir.M()));
    uint shiftedComponentType = neighborPrefixReservoir.componentType(); // used for reconnecting to the last prefix vertex
    LastVertexState shiftedLastVertexState = {};
    shiftedLastVertexState.Init(0, shiftedComponentType, 0, 0, 0, 0);
    float neighborJacobian = neighborPrefixReservoir.rcJacobian; // modified to be the shifted jacobian

    float3 fShift = shiftPrefix(currentPrimarySd, currentPrimaryHit, neighborReservoir.initRandomSeed, neighborReservoir.pathFlags.prefixLength(),
                                neighborPrefix.hit, sg, shiftedLastVertexState, neighborPrefixReservoir.needRandomReplay(), jacobianShift, neighborJacobian, woShifted, shiftedPrefixHit, pdfShift, pathFootprint, false);

    fShift *= currentReplayThp;


    uint backShiftedComponentType = currentPrefixReservoir.componentType();
    LastVertexState backShiftedLastVertexState = {};
    backShiftedLastVertexState.Init(0, backShiftedComponentType, 0, 0, 0, 0);

    HitInfo backShiftedPrefixHit;

    float pdfBackShift;
    float3 woBackShifted;
    float jacobianBackShift = 1.f;
    float pathFootprintDummy = 0;
    float currentJacobian = currentPrefixReservoir.rcJacobian;

    float3 fBackShift = shiftPrefix(neighborPrimarySd, neighborPrimaryHit, reservoir.initRandomSeed, reservoir.pathFlags.prefixLength(), currentPrefix.hit, sg,
                                    backShiftedLastVertexState, currentPrefixReservoir.needRandomReplay(), jacobianBackShift, currentJacobian, woBackShifted, backShiftedPrefixHit, pdfBackShift, pathFootprintDummy, true);

    fBackShift *= neighborReplayThp;

    float pairwiseK = 1.f;
    float neighborMisWeight = computeNeighborPairwiseMISWeight(fShift, neighborPrefixReservoir.pHat, jacobianShift, pairwiseK, currentPrefixReservoir.M(), neighborPrefixReservoir.M());
    float canonicalMisWeight = computeCanonicalPairwiseMISWeight(currentPrefixReservoir.pHat, fBackShift, jacobianBackShift, pairwiseK, currentPrefixReservoir.M(), neighborPrefixReservoir.M());

    if (!isNeighborValid)
    {
        neighborMisWeight = 0.f;
        canonicalMisWeight = 1.f;
    }

    float w = currentPrefixReservoir.pHat * currentPrefixReservoir.W * canonicalMisWeight;
    w = (isnan(w) || isinf(w)) ? 0.0f : w;
    currentPrefixReservoir.W = w;

    // stream neighbor
    w = luminance(fShift) * neighborPrefixReservoir.W * neighborMisWeight * jacobianShift;
    w = (isnan(w) || isinf(w)) ? 0.0f : w;

    currentPrefixReservoir.W += w;
    currentPrefixReservoir.increaseM(neighborPrefixReservoir.M());

    // Accept?
    bool selectPrev = (sampleNext1D(sg) * currentPrefixReservoir.W < w); 

    if (selectPrev)
    {
        currentPrefix.hit = shiftedPrefixHit;
        currentPrefixReservoir.setComponentType(shiftedComponentType); //
        currentPrefixReservoir.setNeedRandomReplay(neighborPrefixReservoir.needRandomReplay());
        currentPrefixReservoir.rcJacobian = neighborJacobian;
        currentPrefixReservoir.pHat = luminance(fShift);
        currentPrefixReservoir.pathFootprint = pathFootprint;
        currentPrefix.wo = woShifted;
        reservoir = neighborReservoir;
    }
    else
    {
        pathFootprint = currentPrefixReservoir.pathFootprint;
    }

    currentPrefixReservoir.W /= currentPrefixReservoir.pHat;
    currentPrefixReservoir.W = (isnan(currentPrefixReservoir.W) || isinf(currentPrefixReservoir.W))
                                   ? 0.f
                                   : max(0.f, currentPrefixReservoir.W);

    reservoir.pathFlags.insertUserFlag(selectPrev);
    return selectPrev;
}

