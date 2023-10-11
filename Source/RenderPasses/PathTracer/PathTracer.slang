/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import Rendering.RTXDI.RTXDI;
import Rendering.ConditionalReSTIR.ConditionalReSTIR;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import LoadShadingData;
import ColorType;
import NRDHelpers;
__exported import PathState;
__exported import Params;
#include "Rendering/Lights/EmissiveLightSamplerType.slangh"
import Utils.Math.MathHelpers;
import Utils.Sampling.TemporarySampleGenerator;

/** Path tracer.

    This implements the high-level path tracing logic and is shared among
    different scheduling implementations.
*/
struct PathTracer
{

    /** Interface for querying the closest hit in the scene.
    This is used in `nextHit` to allow using different raytracing APIs.
*/
    interface IClosestHitQuery
    {
        /** Trace a ray against the scene and return the closest hit point.
            Note: The content of the path state must not be changed in the implementation of this interface.
            However, for the ReorderTraceRay scheduler it is necessary to pass the state through to
            TraceRayHitObject as ray payload and then re-materialize it after ray traversal.
            \param[in,out] path Path state.
            \param[in] ray Ray.
            \param[out] hit Hit info (only valid if function returns true).
            \param[out] hitT Hit distance (only valid if function returns true).
            \return Returns true if the ray does intersect the scene.
        */
        [mutating] bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT);
    };

    /** Interface for querying visibility in the scene.
        This is used in `handleHit`.
    */
    interface IVisibilityQuery
    {
        /** Trace a visibility ray against the scene.
            \param[in] ray Ray.
            \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
        */
        [mutating] bool traceVisibilityRay(const Ray ray);
    };

    PathTracerParams params;                        ///< Runtime parameters.

    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.
    ConditionalReSTIR restir;                              ///< Settings for ReSTIR 

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<uint> sampleCount;                    ///< Optional input sample count buffer. Only valid when params.samplesPerPixel == 0.
    Texture2D<uint> sampleOffset;                   ///< Output offset into per-sample buffers. Only valid when params.samplesPerPixel == 0.

    // Outputs
    //RWStructuredBuffer<ColorType> sampleColor;      ///< Output per-sample color if params.samplesPerPixel != 1.
    RWStructuredBuffer<GuideData> sampleGuideData;  ///< Output per-sample guide data.
    NRDBuffers outputNRD;                           ///< Output NRD data.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if params.samplesPerPixel == 1.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseCurves = USE_CURVES;
    static const bool kUseHairMaterial = USE_HAIR_MATERIAL;

    // Additional specialization.
    static const bool kOutputGuideData = OUTPUT_GUIDE_DATA;

    /** Types of samplable lights.
    */
    enum class LightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint    lightType;     ///< Light type this sample comes from (LightType casted to uint).
        uint    triangleId;    ///< for reconstructing hitInfo for sampled emissive lights
        float2  barycentrics; ///< for reconstructing hitInfo for sampled emissive lights
        float   geometryTerm;  ///< for computing jacobian in GRIS

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 normal;      ///< Shading normal at the vertex (zero if not on a surface).
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).

        /** Initializes a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] normal Shading normal.
            \param[in] faceNormal Geometry normal.
        */
        __init(uint index, float3 pos, float3 normal = float3(0.f), float3 faceNormal = float3(0.f))
        {
            this.index = index;
            this.pos = pos;
            this.normal = normal;
            this.faceNormal = faceNormal;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }
    };

    /** Set guiding data when background is hit.
    */
    static void setBackgroundGuideData(inout GuideData guideData, const float3 dir, const float3 Le)
    {
        if (kOutputGuideData)
        {
            guideData.setNormal(-dir);
            // Compress dynamic range similar to UE4.
            const float3 compressedColor = pow(Le / (Le + 1.f), 0.454545f);
            guideData.setAlbedo(compressedColor);
            guideData.setSpecularAlbedo(float3(0.f));
            guideData.setIndirectAlbedo(float3(0.f));
            guideData.setReflectionPos(float3(0.f));
        }
    }

    /** Set guiding data when primary surface is hit.
    */
    static void setPrimarySurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (kOutputGuideData)
        {
            guideData.setNormal(sd.N);
            // Use sum of reflection/transmission albedo as they are denoised together.
            guideData.setAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            guideData.setReflectionPos(sd.posW);

            const float NdotV = saturate(dot(sd.N, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;

            // TODO: Generalize this to arbitrary materials.
            if (sd.mtl.getMaterialType() == MaterialType::Hair)
            {
                // Use a simple heuristic for hair material.
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(float3(0.04f), ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
            else
            {
                // Use pre-integrated FG term for specular
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
        }
    }

    /** Set guiding data when indirect surface is hit.
    */
    static void setIndirectSurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (any(bsdfProperties.emission > 0.f))
        {
            float3 indirectBaseColor = pow(bsdfProperties.emission / (bsdfProperties.emission + 1.0f), 0.454545f);
            guideData.setIndirectAlbedo(indirectBaseColor);
        }
        else
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            guideData.setIndirectAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
        }
        guideData.setReflectionPos(sd.posW);
    }

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.thp = float3(1.f);

        const uint2 pixel = path.getPixel();

        // Create primary ray.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        // Create sample generator.
        uint maxSpp = params.samplesPerPixel > 0 ? params.samplesPerPixel : kMaxSamplesPerPixel;
        if (params.useConditionalReSTIR) maxSpp = restir.sgCount();
        path.sg = SampleGenerator(pixel, params.seed * maxSpp + path.getSampleIdx());

        // Load the primary hit info from the V-buffer.
        const HitInfo hit = HitInfo(vbuffer[pixel]);

        // If invalid, the path is still active and treated as a miss.
        if (hit.isValid())
        {
            path.setHit(hit);
            path.setVertexIndex(1);
        }

        if (params.useConditionalReSTIR)
        {
            path.restirData = ConditionalReSTIRData();
            path.restirData.pathRcInfo.pathFlags.insertPrefixLength(0);
            path.restirData.pathRcInfo.pathFlags.insertRcVertexLength(restir.kMaximumRcLength);
            path.restirData.pathRcInfo.pathFlags.insertPathTreeLength(0);
            path.restirData.pathRcInfo.pathFlags.insertUserFlag(false); // indicate prefix is not written
            path.restirData.pathRcInfo.rcThroughput = 1.f;

            path.restirData.pathRcInfo.rcHit = HitInfo(); // initialize invalid rcHit

            path.restirData.pathRis.pathFlags.insertPrefixLength(restir.kMaximumRcLength); // setting this in pathRis indicates that prefix length hasn't been determined
        }
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
        if (params.useConditionalReSTIR)
        {
            path.restirData.pathRcInfo.rcThroughput *= weight;
        }
    }

    /** Add radiance to the path contribution.
        \param[in,out] path Path state.
        \param[in] radiance Vertex radiance.
    */
    void addToPathContribution(inout PathState path, const float3 radiance)
    {
        if (!params.useConditionalReSTIR) path.L += path.thp * radiance;
    }

    bool generateScatterRayConditionalReSTIR(ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;

        // sample tangent space dir
        bool valid = false;

        // guiding
        int offset = params.getSampleOffsetAssumeOneSpp(path.getPixel());

        sd.mtl.setLobeMask(1); // if this is non-zero, result.pdf will only return the pdf of the sampled lobe
        valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);

        // result.pdf is the single component PDF
        path.restirData.lastScatterPdf = result.pdf;
        path.restirData.lastScatterWeight = result.weight;

        // get total pdf in a separate call

        uint componentType = result.getComponentType();

        // we need to query pdf of unsampled components
        sd.mtl.setLobeMask(~(1 << componentType));

        result.pdf += mi.evalPdf(sd, result.wo);

        // ReSTIR
        path.setSampledBSDFComponent(componentType);
        path.setRough(result.isLobe(LobeType::Diffuse) || result.isLobe(LobeType::Specular) &&
                                                              mi.getProperties(sd).roughness > restir.settings.specularRoughnessThreshold);

        if (params.useConditionalReSTIR &&
            (!restir.subpathSettings.adaptivePrefixLength && path.getVertexLength() + 1 == restir.minimumPrefixLength
             ||
             restir.subpathSettings.adaptivePrefixLength && path.getVertexLength() + 1 >= restir.minimumPrefixLength &&
                 path.isRough() && path.restirData.pathRcInfo.pathFlags.prefixLength() == 0 && !path.restirData.pathRcInfo.pathFlags.isUserFlagSet())) // indicates non-initialized
        {
            path.restirData.pathRcInfo.pathFlags.insertPrefixLength(path.getVertexLength() + 1);
        }

        return generateScatterRayConditionalReSTIR(result, sd, mi, path, valid);
    }

    // params.useConditionalReSTIR
    bool generateScatterRayConditionalReSTIR(const BSDFSample result, const ShadingData sd, const IMaterialInstance mi, inout PathState path, bool valid)
    {
        if (valid) valid = generateScatterRay(result, sd, mi, path);

        // Ignore valid on purpose for now.
        if (kOutputNRDData)
        {
            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool hasDeltaTransmissionLobe = (lobeTypes & (uint)LobeType::DeltaTransmission) != 0;
            const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

            if (path.getVertexIndex() == 1)
            {
                path.setDiffusePrimaryHit(result.isLobe(LobeType::Diffuse));
                path.setSpecularPrimaryHit(result.isLobe(LobeType::Specular));

                if (kOutputNRDAdditionalData)
                {
                    // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
                    path.setDeltaOnlyPath(result.isLobe(LobeType::DeltaReflection) || result.isLobe(LobeType::DeltaTransmission));

                    path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType::DeltaReflection));
                    path.setDeltaTransmissionPath(result.isLobe(LobeType::DeltaTransmission));
                }
            }

            if (path.getVertexIndex() > 1)
            {
                if (hasNonDeltaLobes) path.setDeltaOnlyPath(false);

                if (kOutputNRDAdditionalData && path.isDeltaTransmissionPath() && path.isDeltaOnlyPath() && hasDeltaTransmissionLobe)
                {
                    if (result.isLobe(LobeType::DeltaReflection) && !isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
                    {
                        path.setDeltaTransmissionPath(false);
                    }
                }
            }
        }

        return valid;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;
        bool valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);

        path.setRough(result.isLobe(LobeType::Diffuse) || result.isLobe(LobeType::Specular) &&
                                                              mi.getProperties(sd).roughness > params.specularRoughnessThreshold); 
        // TODO: unify the roughness threshold

        if (valid) valid = generateScatterRay(result, sd, mi, path);

        // Ignore valid on purpose for now.
        if (kOutputNRDData)
        {
            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool hasDeltaTransmissionLobe = (lobeTypes & (uint)LobeType::DeltaTransmission) != 0;
            const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

            if (path.getVertexIndex() == 1)
            {
                path.setDiffusePrimaryHit(result.isLobe(LobeType::Diffuse));
                path.setSpecularPrimaryHit(result.isLobe(LobeType::Specular));

                if (kOutputNRDAdditionalData)
                {
                    // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
                    path.setDeltaOnlyPath(result.isLobe(LobeType::DeltaReflection) || result.isLobe(LobeType::DeltaTransmission));

                    path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType::DeltaReflection));
                    path.setDeltaTransmissionPath(result.isLobe(LobeType::DeltaTransmission));
                }
            }

            if (path.getVertexIndex() > 1)
            {
                if (hasNonDeltaLobes) path.setDeltaOnlyPath(false);

                if (kOutputNRDAdditionalData && path.isDeltaTransmissionPath() && path.isDeltaOnlyPath() && hasDeltaTransmissionLobe)
                {
                    if (result.isLobe(LobeType::DeltaReflection) && !isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
                    {
                        path.setDeltaTransmissionPath(false);
                    }
                }
            }
        }

        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (path.hit.getType() == HitType::Curve);
        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        path.dir = bs.wo;
        updatePathThroughput(path, bs.weight);
        path.pdf = bs.pdf;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = mi.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > params.specularRoughnessThreshold;

            if (isDiffuse)
            {
                path.incrementBounces(BounceType::Diffuse);
            }
            else
            {
                path.incrementBounces(BounceType::Specular);
                path.setSpecular();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            path.incrementBounces(BounceType::Transmission);
            path.setTransmission();

            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }
        }

        // Save the shading normal. This is needed for MIS.
        path.normal = sd.N;

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);
        //print("path.thp", path.thp);

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere (w.r.t. shading normal) should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.normal, upperHemisphere, sg, tls, false)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);
        // params.useConditionalReSTIR
        ls.barycentrics = tls.uv;
        ls.triangleId = tls.triangleIndex;
        ls.geometryTerm = tls.geometryTerm;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        The upper/lower hemisphere is defined as the union of the hemispheres w.r.t. to the shading and face normals.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float cosTheta = dot(vertex.normal, ls.dir);
        // Flip the face normal to point in the same hemisphere as the shading normal.
        float3 faceNormal = sign(dot(vertex.normal, vertex.faceNormal)) * vertex.faceNormal;
        float cosThetaFace = dot(faceNormal, ls.dir);
        if (!sampleUpperHemisphere && (max(cosTheta, cosThetaFace) >= -kMinCosTheta)) return false;
        if (!sampleLowerHemisphere && (min(cosTheta, cosThetaFace) <= kMinCosTheta)) return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    void analyticLightSampleSkipRandomNumber(inout SampleGenerator sg)
    {
        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return;
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        LightData light = gScene.getLight(lightIndex);

        // case LightType::Rect:
        // case LightType::Sphere:
        // case LightType::Disc:
        // case LightType::Distant:
        if ((uint)light.type > 1)
        {
            sampleNext2D(sg);
        }
    }

    void skipLightSampleRandomNumbers(inout SampleGenerator sg)
    {
        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType::EnvMap)
        {
            if (!kUseEnvLight) valid = false;
            else sampleNext2D(sg);
        }
        if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
        {
            if (!kUseEmissiveLights) valid = false;
            else
            {
                sampleNext3D(sg);
                // power sampler uses an additional random number due to alias table
#if _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_POWER
                sampleNext1D(sg);
#endif
            }
        }
        if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
        {
            analyticLightSampleSkipRandomNumber(sg);
        }
    }

    uint getCurrentSeed(SampleGenerator sg)
    {
#if defined(SAMPLE_GENERATOR_TYPE) && SAMPLE_GENERATOR_TYPE == SAMPLE_GENERATOR_TINY_UNIFORM
        return sg.rng.state;
#else
        return 0;
#endif
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeNewRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = luminance(path.thp / (params.useConditionalReSTIR ? path.restirData.rrPdf : 1.f));
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            path.terminate();
            return true;
        }
        if (params.useConditionalReSTIR)
            path.restirData.rrPdf *= 1.f - prob;
        else
            path.thp /= 1.f - prob;
        return false;
    }

    /** Helper to create a texture sampler instance.
        The method for computing texture level-of-detail depends on the configuration.
        \param[in] path Path state.
        \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
        \return Texture sampler instance.
    */
    ITextureSampler createTextureSampler(const PathState path, bool isPrimaryHit, bool isTriangleHit)
    {
        if (kPrimaryLodMode == TexLODMode::RayDiffs && isPrimaryHit && isTriangleHit)
        {
            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return ExplicitGradientTextureSampler(ddx, ddy);
        }
        else
        {
            float lod = isPrimaryHit ? 0.f : params.lodBias;
            return ExplicitLodTextureSampler(lod);
        }
    }

    void writePrefixPathReservoir(inout PathState path)
    {
        uint offset = params.getSampleOffsetAssumeOneSpp(path.getPixel());

        // write stuff to prefix reservoir
        restir.prefixPathReservoirs[offset] = 0.f;

        restir.prefixGBuffer[offset] = PrefixGBuffer();
        restir.prefixReservoirs[offset] = PrefixReservoir(0, false, 0, 0, 0, 0, 1);

        path.restirData.pathRis = PathRisState();
    }

    void writePrefixPathReservoir(inout PathState path,

                                  HitInfo currentHit,

                                  LastVertexState lastVertexState,
                                  float prefixRcJacobian,
                                  float pathFootprint)
    {
        uint offset = params.getSampleOffsetAssumeOneSpp(path.getPixel());

        PrefixGBuffer pgb = PrefixGBuffer(currentHit, -path.dir);
        //
        bool prefixNeedRandomReplay = !restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Reconnection &&
                                        !(lastVertexState.isLastVertexRough() && lastVertexState.isCurrentVertexFarFromPrev());
        PrefixReservoir prefixReservoir = PrefixReservoir(lastVertexState.lastBSDFComponent(), prefixNeedRandomReplay, prefixRcJacobian, luminance(path.thp), pathFootprint);

        restir.prefixGBuffer[offset] = pgb;
        restir.prefixReservoirs[offset] = prefixReservoir;

        // store prefix throughput in a separate buffer
        PrefixInfo prefixInfo;
        prefixInfo.thp = path.thp;
        restir.prefixThroughputs[offset] = prefixInfo;

        // write stuff to prefix reservoir (might just be empty)
        restir.prefixPathReservoirs[offset] = 0.f;

        // reset stuff
        path.restirData.pathRis = PathRisState();
        path.restirData.pathRcInfo.pathFlags.insertRcVertexLength(restir.kMaximumRcLength);
        path.restirData.pathRcInfo.pathFlags.insertDeltaTransmissionBSDFEvent(false, false, 0, true); // clean this
        path.restirData.pathRcInfo.pathFlags.insertDeltaTransmissionBSDFEvent(false, false, 0, false); // clean this
        path.thp = 1.f;
        path.restirData.pathRcInfo.rcThroughput = 1.f;
        path.restirData.pathRcInfo.pathFlags.insertUserFlag(true); // indicate prefix is written
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

        float uRussianRoulette = 0.f;
        if (kUseRussianRoulette) uRussianRoulette = sampleNext1D(path.sg);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);

            float3 absorption = HomogeneousVolumeSampler::evalTransmittance(hvd, length(sd.posW - path.origin));
            if (params.useConditionalReSTIR) path.restirData.lastScatterWeight *= absorption;
            updatePathThroughput(path, absorption);
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create BSDF instance and query its properties.
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        if (params.disableDirectIllumination() && path.getVertexIndex() <= 2) computeEmissive = false;
        if (params.disableGeneralizedDirectIllumination() && !path.allowLighting()) computeEmissive = false;

        // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
        // Skip any additional contribution on the secondary hit unless it comes from a scatter event
        // that RTXDI cannot handle, such as transmission or delta scattering events.
        if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEmissive = false;

        float3 attenuatedEmission = 0.f;

        // Determine if BSDF has non-delta lobes.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        bool isVertexClassifiedAsRoughForNEE = false;
        {
            const bool hasDiffuseLobe = (lobeTypes & (uint)LobeType::Diffuse) != 0;
            const bool highRoughness = bsdfProperties.roughness > restir.settings.specularRoughnessThreshold;
            isVertexClassifiedAsRoughForNEE = hasDiffuseLobe || highRoughness;
        }

        HitInfo currentHit = path.hit;

        // params.useConditionalReSTIR stuff
        float distanceToPrevVertex = length(sd.posW - path.origin);
        float geometryFactor = abs(dot(sd.faceN, sd.V)) / (distanceToPrevVertex * distanceToPrevVertex);
        LastVertexState lastVertexState;
        lastVertexState.Init(distanceToPrevVertex > restir.settings.nearFieldDistanceThreshold * restir.sceneRadius, path.getSampledBSDFComponent(),
                             path.isDelta(), path.isTransmission(), path.isRough(), isVertexClassifiedAsRoughForNEE);
        float lastScatterPdf = path.restirData.lastScatterPdf;
        bool hasRISForBSDF = false;
        //
        float lightPdf = 0.f;
        float misWeight = 1.f;

        if (computeEmissive && any(bsdfProperties.emission > 0.f))
        {
            if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.
                // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleHit triangleHit = path.hit.getTriangleHit();
                TriangleLightHit hit;
                hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.posW = sd.posW;
                hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            addToPathContribution(path, misWeight * bsdfProperties.emission);

            attenuatedEmission = path.thp * misWeight * bsdfProperties.emission;

            // pathRis.pathFlags.prefixLength() == 0 -> prefixLength is written in pathRcInfo, but prefix has not ended yet  
            int prefixLength = !path.restirData.pathRcInfo.pathFlags.isUserFlagSet() ? 0 : path.restirData.pathRcInfo.pathFlags.prefixLength();

            bool isLengthGood = path.getVertexLength() > max(0, prefixLength - 1) + 1;

            bool skipStream = !path.restirData.pathRcInfo.pathFlags.isUserFlagSet();
            if (params.useConditionalReSTIR && isLengthGood && skipStream)
                path.restirData.pathRis.integrand += attenuatedEmission / path.restirData.rrPdf;

            if (params.useConditionalReSTIR && isLengthGood && !skipStream)
            {
                hasRISForBSDF = true;
                // convert to area measure in some suffix reuse cases
                bool selected = path.restirData.pathRis.addBSDFVertex(path.sg, path.restirData.pathRcInfo, path.getVertexLength() - 1, attenuatedEmission,
                                                                        path.dir, bsdfProperties.emission, misWeight * bsdfProperties.emission, path.restirData.rrPdf, lightPdf, (uint)LightType::Emissive);
                if (selected && path.restirData.pathRcInfo.pathFlags.rcVertexLength() == restir.kMaximumRcLength) // we don't yet have an rcVertex
                {
                    if (restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Hybrid && lastVertexState.isCurrentVertexFarFromPrev() && lastVertexState.isLastVertexRough() ||
                        restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Reconnection && path.getVertexLength() == prefixLength + 1)
                    {
                        // we found an RC vertex!
                        // set rcVertexLength to current length (this will make rcVertexLength = reseroivr.pathLength + 1)
                        path.restirData.pathRis.markAsRcVertex(path.getVertexIndex() - 1, currentHit, lastVertexState,
                                                                 lightPdf * geometryFactor, (uint)LightType::Emissive, bsdfProperties.emission, float3(0.f),
                                                                 lastScatterPdf * geometryFactor);
                    }
                }
            }
        }

        if (params.useConditionalReSTIR && path.getVertexLength() > 1 && !hasRISForBSDF) sampleNext1D(path.sg);

        bool hasFinished = hasFinishedSurfaceBounces(path);

        path.restirData.pathTotalLength += distanceToPrevVertex;

        path.restirData.pathRcInfo.pathFlags.insertPathTreeLength(path.getVertexLength());

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinished)
        {
            path.terminate();
            return;
        }

        if (params.useConditionalReSTIR && path.getVertexLength() == path.restirData.pathRcInfo.pathFlags.prefixLength() && path.getVertexLength() > 0)
        {
            if (bsdfProperties.roughness <= restir.settings.specularRoughnessThreshold && (lobeTypes & (uint)LobeType::Diffuse) == 0 && // not supporting multilayer material yet
                restir.subpathSettings.avoidSpecularPrefixEndVertex || 
                distanceToPrevVertex <= restir.subpathSettings.shortSegmentThreshold * restir.sceneRadius && restir.subpathSettings.avoidShortPrefixEndSegment)
            {
                // reject prefix proposal
                path.restirData.pathRcInfo.pathFlags.insertPrefixLength(0);
            }
            else
            {
                float footprint = path.restirData.pathTotalLength;

                writePrefixPathReservoir(path, currentHit, lastVertexState, geometryFactor * lastScatterPdf, footprint);

                 // indicate not to trace suffix
                path.terminate();
                return;
            }
        }

        if (params.disableGeneralizedDirectIllumination() && path.isRough())
        {
            // allow direct lighting from next NEE
            path.setAllowLighting(true);
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeNewRayOrigin();
        }

        // Check if we should apply NEE.
        bool applyNEE = kUseNEE && hasNonDeltaLobes;

        // Check if sample from RTXDI should be applied instead of NEE.
        bool applyRTXDI = kUseRTXDI && (isPrimaryHit && !params.disableDirectIllumination()) && hasNonDeltaLobes;

        if (params.disableGeneralizedDirectIllumination() && !path.allowLighting()) 
        {
            applyNEE = false;
            applyRTXDI = false;
        }

        path.setLightSampled(false, false);

        // for params.useConditionalReSTIR
        bool hasRISForNEE = false;

        if (applyNEE && (!isPrimaryHit || !params.disableDirectIllumination()) || applyRTXDI)        
        {
            // for params.useConditionalReSTIR
            // cache (NEE sampling) seed for temporal validation
            uint neeSeed = getCurrentSeed(path.sg);

            LightSample ls = {};
            bool validSample = false;

            if (applyRTXDI)
            {
                // Query final sample from RTXDI.
                validSample = gRTXDI.getFinalSample(path.getPixel(), ls.dir, ls.distance, ls.Li);
                ls.origin = path.origin;
            }
            else
            {
                // Setup path vertex.
                PathVertex vertex = PathVertex(path.getVertexIndex(), sd.posW, sd.N, sd.faceN);

                // Determine if upper/lower hemispheres need to be sampled.
                bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
                if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);

                // Sample a light.
                validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
                path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
            }

            if (validSample)
            {
                // for params.useConditionalReSTIR
                if (ls.lightType == (uint)LightType::Analytic && ls.pdf == 0.f)
                    ls.pdf = getAnalyicSelectionProbability(); // analytic light doesn't have a solid angle PDF.
                float3 Le = ls.Li * ls.pdf;

                // Apply MIS weight.
                if (kUseMIS && !applyRTXDI && ls.lightType != (uint)LightType::Analytic)
                {
                    float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
                    ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
                }

                float3 weight = mi.eval(sd, ls.dir, path.sg);
                float3 Lr = weight * ls.Li;
                if (any(Lr > 0.f))
                {
                    Ray ray = ls.getVisibilityRay();

                    if (isCurvePolyTubeHit)
                    {
                        // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                        // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                        if (dot(sd.N, ray.dir) < 0.f)
                        {
                            ray.origin = ray.origin - sd.N * sd.curveRadius * 2.1f;
                        }
                    }

                    if (!applyNEE || path.getSampleIdx() == 0)
                        logTraceRay(PixelStatsRayType::Visibility);
                    bool visible = vq.traceVisibilityRay(ray);
                    if (visible)
                    {
                        addToPathContribution(path, Lr);

                        if (params.useConditionalReSTIR && isPrimaryHit && applyRTXDI)
                        {
                            float factor = !applyRTXDI ? 1.f / restir.subpathSettings.numIntegrationPrefixes : 1;
                            path.L += factor * path.thp * Lr;
                        }

                        bool skipStream = !path.restirData.pathRcInfo.pathFlags.isUserFlagSet();
                        if (params.useConditionalReSTIR && !isPrimaryHit && skipStream)
                            path.restirData.pathRis.integrand += path.thp * Lr / path.restirData.rrPdf; 

                        if (params.useConditionalReSTIR && !isPrimaryHit && !skipStream)
                        {
                            const bool hasDiffuseLobe = (lobeTypes & (uint)LobeType::Diffuse) != 0;
                            const bool highRoughness = bsdfProperties.roughness > restir.settings.specularRoughnessThreshold;
                            bool isNEEVertexClassifiedAsRough = hasDiffuseLobe || highRoughness;
                            // test connectability
                            bool canConnect =
                                restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Reconnection && path.getVertexLength() == path.restirData.pathRcInfo.pathFlags.prefixLength() + 1 ||
                                restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Hybrid &&
                                    path.getVertexLength() >= path.restirData.pathRcInfo.pathFlags.prefixLength() + 1 &&
                                    path.restirData.pathRcInfo.pathFlags.rcVertexLength() == restir.kMaximumRcLength &&
                                    lastVertexState.isCurrentVertexFarFromPrev() && isNEEVertexClassifiedAsRough && lastVertexState.isLastVertexRough();

                            hasRISForNEE = true;

                            bool selected = path.restirData.pathRis.addNeeVertex(path.sg, path.restirData.pathRcInfo, canConnect, path.getVertexLength(),
                                                                                   path.thp * Lr, ls.dir, Le, Lr, path.restirData.rrPdf, ls.pdf, ls.lightType);

                            bool shouldOverwrite = selected && canConnect;

                            if (shouldOverwrite) path.restirData.pathRis.rcHit = currentHit;    
                            path.restirData.pathRis.rcJacobian = shouldOverwrite ? lastScatterPdf * geometryFactor * ls.pdf : path.restirData.pathRis.rcJacobian;

                            // assuming we don't have point lights. Also assume that light emission don't change with direction
                            if (selected && !canConnect && path.restirData.pathRcInfo.pathFlags.rcVertexLength() == restir.kMaximumRcLength)
                            {
                                path.restirData.pathRis.rcJacobian = ls.pdf;
                                path.restirData.pathRis.rcIrrad = Le;
                                path.restirData.pathRis.rcWi = ls.dir;
                                path.restirData.pathRis.pathFlags.insertRcVertexLength(path.restirData.pathRis.pathFlags.pathLength() + 1);
                                if (ls.lightType == (uint)LightType::Emissive)
                                {
                                    TriangleHit neeTriangleHit;
                                    uint2 meshPrimID = gScene.lightCollection.getSceneMeshPrimID(ls.triangleId);
                                    GeometryInstanceID neeGeoInstanceId;
                                    neeGeoInstanceId.index = meshPrimID.x;
                                    neeTriangleHit.instanceID = neeGeoInstanceId;
                                    neeTriangleHit.primitiveIndex = meshPrimID.y;
                                    neeTriangleHit.barycentrics = ls.barycentrics;

                                    HitInfo neeHitInfo = HitInfo(neeTriangleHit);
                                    
                                    path.restirData.pathRis.rcHit = neeHitInfo;
                                    
                                    path.restirData.pathRis.rcJacobian *= ls.geometryTerm;
                                    path.restirData.pathRis.lightPdf *= ls.geometryTerm; // stored in area measure
                                }
                            }
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
                            path.restirData.pathRis.rcRandomSeed = shouldOverwrite ? neeSeed : path.restirData.pathRis.rcRandomSeed;
#endif                            
                            path.restirData.pathRis.pathFlags.tryInsertBounceTypesBeforeRcVertex(lastVertexState.data & 0xf, shouldOverwrite);
                        }
                    }
                }
            }
        }
        else if (params.useConditionalReSTIR && !isPrimaryHit)
        {
            skipLightSampleRandomNumbers(path.sg);
        }

        // make sure that random numbers match
        if (params.useConditionalReSTIR && !isPrimaryHit && !hasRISForNEE) sampleNext1D(path.sg);

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, kUseRussianRoulette)) return;
        }

        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();

        // Generate the next path segment or terminate.
        bool valid;

        uint tempRandomSeed = getCurrentSeed(path.sg);

        if (params.useConditionalReSTIR)
        {
            // records initial random seed for VertexIndex = 1
            // records suffixRandomSeed when reaching prefix length
            valid = generateScatterRayConditionalReSTIR(sd, mi, path);                
        }
        else
        {
            valid = generateScatterRay(sd, mi, path);
        }

        if (params.useConditionalReSTIR)
        {
            if (path.getVertexLength() == 0) path.restirData.pathRcInfo.initRandomSeed = tempRandomSeed;
            if (path.getVertexLength() == path.restirData.pathRcInfo.pathFlags.prefixLength())
                path.restirData.pathRcInfo.suffixInitRandomSeed = tempRandomSeed;
        }

        if (params.useConditionalReSTIR && path.getVertexLength() > 0)
        {
            // can this be the rcVertex for the prefix?
            bool isCurrentVertexClassifiedAsRough = path.isRough();
            // test connectability
            bool canConnect = restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Reconnection && path.getVertexLength() == path.restirData.pathRcInfo.pathFlags.prefixLength() + 1 ||
                              restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Hybrid &&
                                  path.getVertexLength() >= path.restirData.pathRcInfo.pathFlags.prefixLength() + 1 &&
                                  path.restirData.pathRcInfo.pathFlags.rcVertexLength() == restir.kMaximumRcLength &&
                                  lastVertexState.isCurrentVertexFarFromPrev() && isCurrentVertexClassifiedAsRough && lastVertexState.isLastVertexRough();
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
            path.restirData.pathRcInfo.rcRandomSeed = canConnect ? getCurrentSeed(path.sg) : path.restirData.pathRcInfo.rcRandomSeed;
#endif            
            path.restirData.pathRcInfo.rcWi = canConnect ? path.dir : path.restirData.pathRcInfo.rcWi;
            path.restirData.pathRcInfo.pathFlags.insertRcVertexLength(canConnect ? path.getVertexIndex() - 1 : path.restirData.pathRcInfo.pathFlags.rcVertexLength());
            if (canConnect) path.restirData.pathRcInfo.rcHit = currentHit;
            path.restirData.pathRcInfo.pathFlags.tryInsertBounceTypesBeforeRcVertex(lastVertexState.data & 0xf, canConnect);
            path.restirData.pathRcInfo.rcJacobian = canConnect ? lastScatterPdf * path.restirData.lastScatterPdf * geometryFactor : path.restirData.pathRcInfo.rcJacobian;
            path.restirData.pathRcInfo.pathFlags.tryInsertBounceTypesAfterRcVertex(uint(path.isDelta()) | uint(path.isTransmission()) << 1 | uint(path.getSampledBSDFComponent()) << 2, canConnect);
            path.restirData.pathRcInfo.rcThroughput = canConnect ? float3(1.f) : path.restirData.pathRcInfo.rcThroughput;
        }

        // Output guide data.
        if (path.getVertexIndex() == 1)
        {
            setPrimarySurfaceGuideData(path.guideData, sd, bsdfProperties);
        }
        if (path.getVertexIndex() == 2 && (path.getBounces(BounceType::Specular) == 1 || path.getBounces(BounceType::Transmission) == 1))
        {
            setIndirectSurfaceGuideData(path.guideData, sd, bsdfProperties);
        }

        if (kOutputNRDData)
        {
            const uint2 pixel = path.getPixel();
            const uint outSampleIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

            setNRDPrimaryHitEmission(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, attenuatedEmission);
            setNRDPrimaryHitReflectance(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, sd, bsdfProperties);
            setNRDSampleHitDist(outputNRD, path, outSampleIdx); // if ReSTIR end up selecting a sample, this will be overwritten
            setNRDSampleReflectance(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, sd, bsdfProperties, lobeTypes, wasDeltaOnlyPathBeforeScattering);
            setNRDSampleEmission(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, attenuatedEmission, wasDeltaOnlyPathBeforeScattering);

            // this should be RTXDI contribution
            setNRDPrimaryHitSeparatedRadiance(outputNRD, kUseNRDDemodulation, path, isPrimaryHit, outSampleIdx);
        }

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        if (params.disableDirectIllumination() && path.getVertexIndex() <= 2) computeEnv = false;
        if (params.disableGeneralizedDirectIllumination() && !path.allowLighting()) computeEnv = false;

            // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
        // Skip any additional contribution on the secondary hit unless it comes from a scatter event
        // that RTXDI cannot handle, such as transmission, delta or volume scattering events.
        if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEnv = false;

        float3 emitterRadiance = 0.f;

        if (computeEnv)
        {
            logPathVertex();

            float misWeight = 1.f;
            float lightPdf = 0.f;

            if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            float3 Le = envMapSampler.eval(path.dir);
            emitterRadiance = misWeight * Le;
            addToPathContribution(path, emitterRadiance);

            int prefixLength = !path.restirData.pathRcInfo.pathFlags.isUserFlagSet() ? 0 : path.restirData.pathRcInfo.pathFlags.prefixLength();

            bool skipStream = !path.restirData.pathRcInfo.pathFlags.isUserFlagSet();

            bool isLengthGood = path.getVertexLength() > max(0, prefixLength - 1) + 1;

            if (params.useConditionalReSTIR && isLengthGood && skipStream)
                path.restirData.pathRis.integrand += path.thp / path.restirData.rrPdf * emitterRadiance;

            if (params.useConditionalReSTIR && isLengthGood && !skipStream)
            {
                LastVertexState lastVertexState;
                lastVertexState.Init(true, path.getSampledBSDFComponent(), path.isDelta(), path.isTransmission(), path.isRough(), false);

                bool selected = path.restirData.pathRis.addBSDFVertex(path.sg, path.restirData.pathRcInfo, path.getVertexLength() - 1, 
                                                                        path.thp * emitterRadiance, path.dir, Le, emitterRadiance, path.restirData.rrPdf, lightPdf, (uint)LightType::EnvMap);

                if (selected && path.restirData.pathRcInfo.pathFlags.rcVertexLength() == restir.kMaximumRcLength)
                {
                    if (restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Reconnection && path.getVertexLength() == prefixLength + 1 ||
                        restir.shiftMapping == (uint)ConditionalReSTIR::ShiftMapping::Hybrid && path.isRough())
                    {
                        // we found an RC vertex!
                        // set rcVertexLength to current length (this will make rcVertexLength = reseroivr.pathLength + 1)
                        HitInfo dummyHit = HitInfo();
                        
                        path.restirData.pathRis.markAsRcVertex(path.getVertexIndex() - 1, dummyHit, lastVertexState,
                                                                 lightPdf, (uint)LightType::EnvMap, Le, path.dir, path.restirData.lastScatterPdf);
                    }
                }
            }

            if (kOutputGuideData && path.getVertexIndex() == 2
                && (path.getBounces(BounceType::Specular) == 1
                || path.getBounces(BounceType::Transmission) == 1))
            {
                // Compress dynamic range similar to UE4.
                float3 compressedColor = pow(Le / (Le + 1.0f), 0.454545f);
                path.guideData.setIndirectAlbedo(compressedColor);
                path.guideData.setReflectionPos(path.dir * kEnvMapDepth);
            }
        }

        if (kOutputGuideData && path.getVertexIndex() == 1)
        {
            path.guideData.setNormal(-path.dir);
        }

        if (kOutputNRDData && !params.useConditionalReSTIR)
        {
            const uint outSampleIdx = params.getSampleOffset(path.getPixel(), sampleOffset) + path.getSampleIdx();
            setNRDSampleHitDist(outputNRD, path, outSampleIdx);
        }

#if defined(DELTA_REFLECTION_PASS)
        if (path.isDeltaReflectionPrimaryHit())
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
        }
        else
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
        }
#elif defined(DELTA_TRANSMISSION_PASS)
        if (path.isDeltaTransmissionPath())
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, path.origin + path.dir * kNRDInvalidPathLength);
        }
        else
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, 0.f);
        }
#endif

        path.terminate();
    }

    void nextHit<ClosestHitQuery : IClosestHitQuery>(inout PathState path, inout ClosestHitQuery chq)
    {
        Ray ray = path.getScatterRay();
        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray.
        logTraceRay(PixelStatsRayType::ClosestHit);
        HitInfo hit;
        float hitT;

        if (!chq.traceRay(path, ray, hit, hitT)) hit = {};

        if (hit.isValid())
        {
            path.sceneLength += float16_t(hitT);
            
            path.setHit(hit);

            if (!path.interiorList.isEmpty())
            {
                const uint materialID = path.interiorList.getTopMaterialID();
                const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
                updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(hvd, hitT));
            }
        }
        else
        {
            path.sceneLength += float16_t(kNRDInvalidPathLength);
            path.clearHit();
        }
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(PathState path)
    {
        assert(!any(isnan(path.L)));

        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        const uint outIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

        if (params.useConditionalReSTIR)
        {
            uint reservoirIdx = params.getSampleOffsetAssumeOneSpp(pixel);

            path.restirData.pathRis.pathFlags.insertPathTreeLength(path.restirData.pathRcInfo.pathFlags.pathTreeLength());
            path.restirData.pathRis.pathFlags.insertUserFlag(false); // indicate current frame
            // in this case, we haven't found a prefix
            if (!path.restirData.pathRcInfo.pathFlags.isUserFlagSet())        // TODO: this might not be necessary since we don't need to explicitly add prefixes within prefix length at the end now after NEE refactor
                path.restirData.pathRis.pathFlags.insertPrefixLength(kMaxSurfaceBounces); // TODO: consider remove
            else
                path.restirData.pathRis.pathFlags.insertPrefixLength(min(path.restirData.pathRcInfo.pathFlags.prefixLength(), kMaxSurfaceBounces));

            uint totalSamples = params.samplesPerPixel;
            if (params.samplesPerPixel == 0)
            {
                totalSamples = sampleCount[pixel];
            }

            PathReservoir pathReservoir;
            
            if (path.getSampleIdx() == 0)
            {
                pathReservoir = PathReservoir::createFromRisState(path.restirData.pathRis, path.restirData.pathRcInfo.initRandomSeed, path.restirData.pathRcInfo.suffixInitRandomSeed);
            }
            else
            {
                pathReservoir = restir.pathReservoirs[reservoirIdx];
                pathReservoir.add(path.restirData.pathRis, path.restirData.pathRcInfo.initRandomSeed, path.restirData.pathRcInfo.suffixInitRandomSeed, path.sg);
            }

            // // converting from w_sum to W
            if (path.getSampleIdx() == totalSamples - 1)
            {
                pathReservoir.finalizeGRISDivideByM();
                pathReservoir.M = 1;
            }

            restir.pathReservoirs[reservoirIdx] = pathReservoir;
        }

        float3 color = path.L;

        if (any(isnan(color) || isinf(color))) color = 0.f;

        if (path.getSampleIdx() == 0)
        {
            // Write color directly to frame buffer.
            outputColor[pixel] = float4(color / params.samplesPerPixel, 1.f);
        }
        else
        {
            // Write color to per-sample buffer.
            outputColor[pixel] += float4(color / params.samplesPerPixel, 1.f);
        }

        if (kOutputGuideData)
        {
            sampleGuideData[outIdx] = path.guideData;
        }

        if (kOutputNRDData)
        {
            // TODO: Optimize this for 1 SPP. It doesn't have to go through resolve pass like the color above.
            NRDRadiance data = {};

            if (path.isDiffusePrimaryHit()) data.setPathType(NRDPathType::Diffuse);
            else if (path.isSpecularPrimaryHit()) data.setPathType(NRDPathType::Specular);
            else if (path.isDeltaReflectionPrimaryHit()) data.setPathType(NRDPathType::DeltaReflection);
            else if (path.isDeltaTransmissionPath()) data.setPathType(NRDPathType::DeltaTransmission);
            else data.setPathType(NRDPathType::Residual);

            data.setRadiance(path.L);

            outputNRD.sampleRadiance[outIdx] = data;
        }
    }
}
