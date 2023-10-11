/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#include "PathTracer.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

const RenderPass::Info PathTracer::kInfo { "PathTracer", "Reference path tracer." };

namespace
{
    const std::string kGeneratePathsFilename = "RenderPasses/PathTracer/GeneratePaths.cs.slang";
    const std::string kTracePassFilename = "RenderPasses/PathTracer/TracePass.cs.slang";
    const std::string kResolvePassFilename = "RenderPasses/PathTracer/ResolvePass.cs.slang";
    const std::string kReflectTypesFile = "RenderPasses/PathTracer/ReflectTypes.cs.slang";

    const std::string kShaderModel = "6_5";

    // Render pass inputs and outputs.
    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "mvec";
    const std::string kInputViewDir = "viewW";
    const std::string kInputSampleCount = "sampleCount";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */ },
        { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
        { kInputSampleCount,    "gSampleCount",     "Sample count buffer (integer format)", true /* optional */, ResourceFormat::R8Uint },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputSubColor = "subColor";
    const std::string kOutputVariance = "variance";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputSpecularAlbedo = "specularAlbedo";
    const std::string kOutputIndirectAlbedo = "indirectAlbedo";
    const std::string kOutputNormal = "normal";
    const std::string kOutputReflectionPosW = "reflectionPosW";
    const std::string kOutputRayCount = "rayCount";
    const std::string kOutputPathLength = "pathLength";
    const std::string kOutputNRDDiffuseRadianceHitDist = "nrdDiffuseRadianceHitDist";
    const std::string kOutputNRDSpecularRadianceHitDist = "nrdSpecularRadianceHitDist";
    const std::string kOutputNRDEmission = "nrdEmission";
    const std::string kOutputNRDDiffuseReflectance = "nrdDiffuseReflectance";
    const std::string kOutputNRDSpecularReflectance = "nrdSpecularReflectance";
    const std::string kOutputNRDDeltaReflectionRadianceHitDist = "nrdDeltaReflectionRadianceHitDist";
    const std::string kOutputNRDDeltaReflectionReflectance = "nrdDeltaReflectionReflectance";
    const std::string kOutputNRDDeltaReflectionEmission = "nrdDeltaReflectionEmission";
    const std::string kOutputNRDDeltaReflectionNormWRoughMaterialID = "nrdDeltaReflectionNormWRoughMaterialID";
    const std::string kOutputNRDDeltaReflectionPathLength = "nrdDeltaReflectionPathLength";
    const std::string kOutputNRDDeltaReflectionHitDist = "nrdDeltaReflectionHitDist";
    const std::string kOutputNRDDeltaTransmissionRadianceHitDist = "nrdDeltaTransmissionRadianceHitDist";
    const std::string kOutputNRDDeltaTransmissionReflectance = "nrdDeltaTransmissionReflectance";
    const std::string kOutputNRDDeltaTransmissionEmission = "nrdDeltaTransmissionEmission";
    const std::string kOutputNRDDeltaTransmissionNormWRoughMaterialID = "nrdDeltaTransmissionNormWRoughMaterialID";
    const std::string kOutputNRDDeltaTransmissionPathLength = "nrdDeltaTransmissionPathLength";
    const std::string kOutputNRDDeltaTransmissionPosW = "nrdDeltaTransmissionPosW";
    const std::string kOutputNRDResidualRadianceHitDist = "nrdResidualRadianceHitDist";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,                                     "",     "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputSubColor, "", "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float},
        { kOutputVariance, "", "Output variance (avg X^2, avg X, var estimate)", true /* optional */, ResourceFormat::RGBA32Float},
        { kOutputAlbedo,                                    "",     "Output albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputSpecularAlbedo,                            "",     "Output specular albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputIndirectAlbedo,                            "",     "Output indirect albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputNormal,                                    "",     "Output normal (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputReflectionPosW,                            "",     "Output reflection pos (world space)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputRayCount,                                  "",     "Per-pixel ray count", true /* optional */, ResourceFormat::R32Uint },
        { kOutputPathLength,                                "",     "Per-pixel path length", true /* optional */, ResourceFormat::R32Uint },
        // NRD outputs
        { kOutputNRDDiffuseRadianceHitDist,                 "",     "Output demodulated diffuse color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDSpecularRadianceHitDist,                "",     "Output demodulated specular color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDEmission,                               "",     "Output primary surface emission", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDiffuseReflectance,                     "",     "Output primary surface diffuse reflectance", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputNRDSpecularReflectance,                    "",     "Output primary surface specular reflectance", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputNRDDeltaReflectionRadianceHitDist,         "",     "Output demodulated delta reflection color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDeltaReflectionReflectance,             "",     "Output delta reflection reflectance color (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputNRDDeltaReflectionEmission,                "",     "Output delta reflection emission color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDeltaReflectionNormWRoughMaterialID,    "",     "Output delta reflection world normal, roughness, and material ID", true /* optional */, ResourceFormat::RGB10A2Unorm },
        { kOutputNRDDeltaReflectionPathLength,              "",     "Output delta reflection path length", true /* optional */, ResourceFormat::R16Float },
        { kOutputNRDDeltaReflectionHitDist,                 "",     "Output delta reflection hit distance", true /* optional */, ResourceFormat::R16Float },
        { kOutputNRDDeltaTransmissionRadianceHitDist,       "",     "Output demodulated delta transmission color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDeltaTransmissionReflectance,           "",     "Output delta transmission reflectance color (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputNRDDeltaTransmissionEmission,              "",     "Output delta transmission emission color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDeltaTransmissionNormWRoughMaterialID,  "",     "Output delta transmission world normal, roughness, and material ID", true /* optional */, ResourceFormat::RGB10A2Unorm },
        { kOutputNRDDeltaTransmissionPathLength,            "",     "Output delta transmission path length", true /* optional */, ResourceFormat::R16Float },
        { kOutputNRDDeltaTransmissionPosW,                  "",     "Output delta transmission position", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDResidualRadianceHitDist,                "",     "Output residual color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
    };

    // UI variables.
    const Gui::DropdownList kColorFormatList =
    {
        { (uint32_t)ColorFormat::RGBA32F, "RGBA32F (128bpp)" },
        { (uint32_t)ColorFormat::LogLuvHDR, "LogLuvHDR (32bpp)" },
    };

    const Gui::DropdownList kMISHeuristicList =
    {
        { (uint32_t)MISHeuristic::Balance, "Balance heuristic" },
        { (uint32_t)MISHeuristic::PowerTwo, "Power heuristic (exp=2)" },
        { (uint32_t)MISHeuristic::PowerExp, "Power heuristic" },
    };

    const Gui::DropdownList kEmissiveSamplerList =
    {
        { (uint32_t)EmissiveLightSamplerType::Uniform, "Uniform" },
        { (uint32_t)EmissiveLightSamplerType::LightBVH, "LightBVH" },
        { (uint32_t)EmissiveLightSamplerType::Power, "Power" },
    };

    const Gui::DropdownList kLODModeList =
    {
        { (uint32_t)TexLODMode::Mip0, "Mip0" },
        { (uint32_t)TexLODMode::RayDiffs, "Ray Diffs" }
    };

    const Gui::DropdownList kRenderModePreset = {
        {0, "CRIS/Conditional ReSTIR"},
        {1, "MMIS"},
        {2, "Path Tracing"}
    };

    const Gui::DropdownList kDIMode = {
        {0, "Enable DI"},
        {1, "Disable DI"},
        {2, "Disable DI through specular chain"},
    };


    // Scripting options.
    const std::string kSamplesPerPixel = "samplesPerPixel";
    const std::string kTotalSamplesPerPixel = "totalSamplesPerPixel";
    const std::string kMaxSurfaceBounces = "maxSurfaceBounces";
    const std::string kMaxDiffuseBounces = "maxDiffuseBounces";
    const std::string kMaxSpecularBounces = "maxSpecularBounces";
    const std::string kMaxTransmissionBounces = "maxTransmissionBounces";

    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kUseBSDFSampling = "useBSDFSampling";
    const std::string kUseRussianRoulette = "useRussianRoulette";
    const std::string kUseLambertianDiffuse = "useLambertianDiffuse";
    const std::string kDisableDirectIllumination = "disableDirectIllumination";
    const std::string kDisableGeneralizedDirectIllumination = "disableGeneralizedDirectIllumination";
    const std::string kDisableDiffuse = "disableDiffuse";
    const std::string kDisableSpecular = "disableSpecular";
    const std::string kDisableTranslucency = "disableTranslucency";

    const std::string kUseNEE = "useNEE";
    const std::string kUseMIS = "useMIS";
    const std::string kMISHeuristic = "misHeuristic";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";
    const std::string kUseRTXDI = "useRTXDI";
    const std::string kRTXDIOptions = "RTXDIOptions";
    const std::string kUseReSTIR = "useConditionalReSTIR";
    const std::string kConditionalReSTIROptions = "ConditionalReSTIROptions";

    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kAdjustShadingNormals = "adjustShadingNormals";
    const std::string kMaxNestedMaterials = "maxNestedMaterials";
    const std::string kUseLightsInDielectricVolumes = "useLightsInDielectricVolumes";
    const std::string kDisableCaustics = "disableCaustics";
    const std::string kSpecularRoughnessThreshold = "specularRoughnessThreshold";
    const std::string kPrimaryLodMode = "primaryLodMode";
    const std::string kLODBias = "lodBias";

    const std::string kOutputSize = "outputSize";
    const std::string kFixedOutputSize = "fixedOutputSize";
    const std::string kColorFormat = "colorFormat";
    const std::string kSeedOffset = "seedOffset";

    const std::string kUseNRDDemodulation = "useNRDDemodulation";
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(PathTracer::kInfo, PathTracer::create);
    ScriptBindings::registerBinding(PathTracer::registerBindings);
    ScriptBindings::registerBinding(ConditionalReSTIRPass::scriptBindings);
}

void PathTracer::registerBindings(pybind11::module& m)
{
    pybind11::enum_<ColorFormat> colorFormat(m, "ColorFormat");
    colorFormat.value("RGBA32F", ColorFormat::RGBA32F);
    colorFormat.value("LogLuvHDR", ColorFormat::LogLuvHDR);

    pybind11::enum_<MISHeuristic> misHeuristic(m, "MISHeuristic");
    misHeuristic.value("Balance", MISHeuristic::Balance);
    misHeuristic.value("PowerTwo", MISHeuristic::PowerTwo);
    misHeuristic.value("PowerExp", MISHeuristic::PowerExp);

    pybind11::class_<PathTracer, RenderPass, PathTracer::SharedPtr> pass(m, "PathTracer");
    pass.def_property_readonly("pixelStats", &PathTracer::getPixelStats);

    pass.def_property("useFixedSeed",
        [](const PathTracer* pt) { return pt->mParams.useFixedSeed ? true : false; },
        [](PathTracer* pt, bool value) { pt->mParams.useFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const PathTracer* pt) { return pt->mParams.fixedSeed; },
        [](PathTracer* pt, uint32_t value) { pt->mParams.fixedSeed = value; }
    );
}

PathTracer::SharedPtr PathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new PathTracer(dict));
}

PathTracer::PathTracer(const Dictionary& dict)
    : RenderPass(kInfo)
{
    if (!gpDevice->isShaderModelSupported(Device::ShaderModel::SM6_5))
    {
        throw RuntimeError("PathTracer: Shader Model 6.5 is not supported by the current device");
    }
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw RuntimeError("PathTracer: Raytracing Tier 1.1 is not supported by the current device");
    }

    parseDictionary(dict);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mStaticParams.sampleGenerator);

    // Create resolve pass. This doesn't depend on the scene so can be created here.
    auto defines = mStaticParams.getDefines(*this);
    mpResolvePass = ComputePass::create(Program::Desc(kResolvePassFilename).setShaderModel(kShaderModel).csEntry("main"), defines, false);

    // Note: The other programs are lazily created in updatePrograms() because a scene needs to be present when creating them.

    mpPixelStats = PixelStats::create();
    mpPixelDebug = PixelDebug::create();
}

void PathTracer::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        // Rendering parameters
        if (key == kSamplesPerPixel) mParams.samplesPerPixel = value;
        else if (key == kMaxSurfaceBounces) mStaticParams.maxSurfaceBounces = value;
        else if (key == kMaxDiffuseBounces) mStaticParams.maxDiffuseBounces = value;
        else if (key == kMaxSpecularBounces) mStaticParams.maxSpecularBounces = value;
        else if (key == kMaxTransmissionBounces) mStaticParams.maxTransmissionBounces = value;

        // Sampling parameters
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mParams.fixedSeed = value; mParams.useFixedSeed = true; }
        else if (key == kUseBSDFSampling) mStaticParams.useBSDFSampling = value;
        else if (key == kUseRussianRoulette) mStaticParams.useRussianRoulette = value;
        else if (key == kUseLambertianDiffuse) mStaticParams.useLambertianDiffuse = value;
        else if (key == kDisableDirectIllumination) mParams.DIMode = value ? 1 : 0;
        // don't use kDisableDirectIllumination and kDisableGeneralizedDirectIllumination in the same script
        else if (key == kDisableGeneralizedDirectIllumination) mParams.DIMode = value ? 2 : 0;
        else if (key == kDisableDiffuse) mStaticParams.disableDiffuse = value;
        else if (key == kDisableSpecular) mStaticParams.disableSpecular = value;
        else if (key == kDisableTranslucency) mStaticParams.disableTranslucency = value;
        else if (key == kUseNEE) mStaticParams.useNEE = value;
        else if (key == kUseMIS) mStaticParams.useMIS = value;
        else if (key == kMISHeuristic) mStaticParams.misHeuristic = value;
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;
        else if (key == kUseRTXDI) mStaticParams.useRTXDI = value;
        else if (key == kUseReSTIR) mParams.useConditionalReSTIR = value;
        else if (key == kRTXDIOptions) mRTXDIOptions = value;
        else if (key == kConditionalReSTIROptions) mConditionalReSTIROptions = value;

        // Material parameters
        else if (key == kUseAlphaTest) mStaticParams.useAlphaTest = value;
        else if (key == kAdjustShadingNormals) mStaticParams.adjustShadingNormals = value;
        else if (key == kMaxNestedMaterials) mStaticParams.maxNestedMaterials = value;
        else if (key == kUseLightsInDielectricVolumes) mStaticParams.useLightsInDielectricVolumes = value;
        else if (key == kDisableCaustics) mStaticParams.disableCaustics = value;
        else if (key == kSpecularRoughnessThreshold) mParams.specularRoughnessThreshold = value;
        else if (key == kPrimaryLodMode) mStaticParams.primaryLodMode = value;
        else if (key == kLODBias) mParams.lodBias = value;

        // Denoising parameters
        else if (key == kUseNRDDemodulation) mStaticParams.useNRDDemodulation = value;

        // Output parameters
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kFixedOutputSize) mFixedOutputSize = value;
        else if (key == kColorFormat) mStaticParams.colorFormat = value;
        else if (key == kSeedOffset) mSeedOffset = value;

        else logWarning("Unknown field '{}' in PathTracer dictionary.", key);
    }

    if (dict.keyExists(kMaxSurfaceBounces))
    {
        // Initialize bounce counts to 'maxSurfaceBounces' if they weren't explicitly set.
        if (!dict.keyExists(kMaxDiffuseBounces)) mStaticParams.maxDiffuseBounces = mStaticParams.maxSurfaceBounces;
        if (!dict.keyExists(kMaxSpecularBounces)) mStaticParams.maxSpecularBounces = mStaticParams.maxSurfaceBounces;
        if (!dict.keyExists(kMaxTransmissionBounces)) mStaticParams.maxTransmissionBounces = mStaticParams.maxSurfaceBounces;
    }
    else
    {
        // Initialize surface bounces.
        mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    }

    bool maxSurfaceBouncesNeedsAdjustment =
        mStaticParams.maxSurfaceBounces < mStaticParams.maxDiffuseBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxSpecularBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxTransmissionBounces;

    // Show a warning if maxSurfaceBounces will be adjusted in validateOptions().
    if (dict.keyExists(kMaxSurfaceBounces) && maxSurfaceBouncesNeedsAdjustment)
    {
        logWarning("'{}' is set lower than '{}', '{}' or '{}' and will be increased.", kMaxSurfaceBounces, kMaxDiffuseBounces, kMaxSpecularBounces, kMaxTransmissionBounces);
    }
}

void PathTracer::validateOptions()
{
    if (mParams.specularRoughnessThreshold < 0.f || mParams.specularRoughnessThreshold > 1.f)
    {
        logWarning("'specularRoughnessThreshold' has invalid value. Clamping to range [0,1].");
        mParams.specularRoughnessThreshold = clamp(mParams.specularRoughnessThreshold, 0.f, 1.f);
    }

    // Static parameters.
    if (mParams.samplesPerPixel < 1 || mParams.samplesPerPixel > kMaxSamplesPerPixel)
    {
        logWarning("'samplesPerPixel' must be in the range [1, {}]. Clamping to this range.", kMaxSamplesPerPixel);
        mParams.samplesPerPixel = std::clamp(mParams.samplesPerPixel, 1, (int)kMaxSamplesPerPixel);
    }

    auto clampBounces = [] (uint32_t& bounces, const std::string& name)
    {
        if (bounces > kMaxBounces)
        {
            logWarning("'{}' exceeds the maximum supported bounces. Clamping to {}.", name, kMaxBounces);
            bounces = kMaxBounces;
        }
    };

    clampBounces(mStaticParams.maxSurfaceBounces, kMaxSurfaceBounces);
    clampBounces(mStaticParams.maxDiffuseBounces, kMaxDiffuseBounces);
    clampBounces(mStaticParams.maxSpecularBounces, kMaxSpecularBounces);
    clampBounces(mStaticParams.maxTransmissionBounces, kMaxTransmissionBounces);

    // Make sure maxSurfaceBounces is at least as many as any of diffuse, specular or transmission.
    uint32_t minSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxSurfaceBounces, minSurfaceBounces);

    if (mStaticParams.primaryLodMode == TexLODMode::RayCones)
    {
        logWarning("Unsupported tex lod mode. Defaulting to Mip0.");
        mStaticParams.primaryLodMode = TexLODMode::Mip0;
    }
}

Dictionary PathTracer::getScriptingDictionary()
{
    if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Dictionary d;

    // Rendering parameters
    d[kSamplesPerPixel] = mParams.samplesPerPixel;
    d[kMaxSurfaceBounces] = mStaticParams.maxSurfaceBounces;
    d[kMaxDiffuseBounces] = mStaticParams.maxDiffuseBounces;
    d[kMaxSpecularBounces] = mStaticParams.maxSpecularBounces;
    d[kMaxTransmissionBounces] = mStaticParams.maxTransmissionBounces;

    // Sampling parameters
    d[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed) d[kFixedSeed] = mParams.fixedSeed;
    d[kUseBSDFSampling] = mStaticParams.useBSDFSampling;
    d[kUseRussianRoulette] = mStaticParams.useRussianRoulette;
    d[kUseLambertianDiffuse] = mStaticParams.useLambertianDiffuse;
    d[kUseNEE] = mStaticParams.useNEE;
    d[kUseMIS] = mStaticParams.useMIS;
    d[kMISHeuristic] = mStaticParams.misHeuristic;
    d[kMISPowerExponent] = mStaticParams.misPowerExponent;
    d[kEmissiveSampler] = mStaticParams.emissiveSampler;
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) d[kLightBVHOptions] = mLightBVHOptions;
    d[kUseRTXDI] = mStaticParams.useRTXDI;
    d[kRTXDIOptions] = mRTXDIOptions;
    d[kUseReSTIR] = mParams.useConditionalReSTIR;
    d[kConditionalReSTIROptions] = mConditionalReSTIROptions;

    // Material parameters
    d[kUseAlphaTest] = mStaticParams.useAlphaTest;
    d[kAdjustShadingNormals] = mStaticParams.adjustShadingNormals;
    d[kMaxNestedMaterials] = mStaticParams.maxNestedMaterials;
    d[kUseLightsInDielectricVolumes] = mStaticParams.useLightsInDielectricVolumes;
    d[kDisableCaustics] = mStaticParams.disableCaustics;
    d[kSpecularRoughnessThreshold] = mParams.specularRoughnessThreshold;
    d[kPrimaryLodMode] = mStaticParams.primaryLodMode;
    d[kLODBias] = mParams.lodBias;

    // Denoising parameters
    d[kUseNRDDemodulation] = mStaticParams.useNRDDemodulation;

    // Output parameters
    d[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed) d[kFixedOutputSize] = mFixedOutputSize;
    d[kColorFormat] = mStaticParams.colorFormat;

    return d;
}

RenderPassReflection PathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void PathTracer::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    auto prevScreenTiles = mParams.screenTiles;

    mParams.frameDim = frameDim;
    if (mParams.frameDim.x > kMaxFrameDimension || mParams.frameDim.y > kMaxFrameDimension)
    {
        throw RuntimeError("Frame dimensions up to {} pixels width/height are supported.", kMaxFrameDimension);
    }

    // Tile dimensions have to be powers-of-two.
    FALCOR_ASSERT(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    FALCOR_ASSERT(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    mParams.screenTiles = div_round_up(mParams.frameDim, kScreenTileDim);

    if (mParams.frameDim != prevFrameDim || mParams.screenTiles != prevScreenTiles)
    {
        mVarsChanged = true;
    }
}

void PathTracer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;

    mParams.frameCount = 0;
    mParams.frameDim = {};
    mParams.screenTiles = {};

    // Need to recreate the RTXDI module when the scene changes.
    mpRTXDI = nullptr;
    mpConditionalReSTIRPass = nullptr;

    // Need to recreate the trace passes because the shader binding table changes.
    mpTracePass = nullptr;
    mpTraceDeltaReflectionPass = nullptr;
    mpTraceDeltaTransmissionPass = nullptr;
    mpGeneratePaths = nullptr;
    mpReflectTypes = nullptr;

    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("PathTracer: This render pass does not support custom primitives.");
        }

        validateOptions();

        mRecompile = true;
    }
}

void PathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (autoCompileMethods && !autoCompileFinished && mpConditionalReSTIRPass)
    {
        if (mRenderModePresetIdPrev != mRenderModePresetId)
            setPresetForMethod(mRenderModePresetId);

        mRenderModePresetIdPrev = mRenderModePresetId;

        if (mWarmupFramesSofar++ > 10) // the second frame
        {
            mWarmupFramesSofar = 0;
            if (mRenderModePresetId == 4)
            {
                autoCompileFinished = true;
                mRenderModePresetId = 0;
                setPresetForMethod(0);
            }
            else
            {
                mRenderModePresetId++;
            }
        }
    }

    if (!beginFrame(pRenderContext, renderData)) return;

    renderData.getDictionary()["freeze"] = mpScene->freeze;

    mUserInteractionRecorder.recordStep(mpScene);

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Prepare the path tracer parameter block.
    // This should be called after all resources have been created.
    preparePathTracer(renderData);

    // Generate paths at primary hits.
    generatePaths(pRenderContext, renderData);

    // Update RTXDI.
    if (mpRTXDI && !mStaticParams.disableDirectIllumination && !mStaticParams.disableGeneralizedDirectIllumination)
    {
        const auto& pMotionVectors = renderData.getTexture(kInputMotionVectors);
        mpRTXDI->update(pRenderContext, pMotionVectors);
    }

    // loop spp times if ReSTIR is enabled

    // Launch separate passes to trace delta reflection and transmission paths to generate respective guide buffers.
    if (mOutputNRDAdditionalData)
    {
        FALCOR_ASSERT(mpTraceDeltaReflectionPass && mpTraceDeltaTransmissionPass);
        tracePass(pRenderContext, renderData, mpTraceDeltaReflectionPass);
        tracePass(pRenderContext, renderData, mpTraceDeltaTransmissionPass);
    }

    uint32_t iters = mParams.useConditionalReSTIR
                         ?  mParams.samplesPerPixel
                         : 1;

    for (uint32_t iter = 0; iter < iters; iter++)
    {
        // Trace pass.
        FALCOR_ASSERT(mpTracePass);
        tracePass(pRenderContext, renderData, mpTracePass, iter);
    }

    if (mpConditionalReSTIRPass && mParams.useConditionalReSTIR)
    {
        mpConditionalReSTIRPass->suffixResamplingPass(
            pRenderContext, renderData.getTexture(kInputVBuffer),
            renderData.getTexture(kInputMotionVectors),
            renderData.getTexture(kOutputColor));
    }

    // Resolve pass.
    resolvePass(pRenderContext, renderData);

    endFrame(pRenderContext, renderData);

    mIsFrozen = mpScene->freeze;

    if (!mpScene->freeze)
    {
        bool shouldFreeze = mUserInteractionRecorder.replayStep(mpScene);
        if (shouldFreeze)
        {
            mpScene->freeze = true;
        }
    }
}

void PathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    auto group = widget.group("User Interaction Recording", false);

    dirty |= mUserInteractionRecorder.renderUI(group);

    if (auto group = widget.group("Rendering Presets", true))
    {
        // method selection
        bool isRenderModeChanged = widget.dropdown("Render Mode Preset", kRenderModePreset, mRenderModePresetId);
        if (isRenderModeChanged && mpConditionalReSTIRPass)
        {
            dirty = true;

            setPresetForMethod(mRenderModePresetId);
        }

        if (mpConditionalReSTIRPass && mParams.useConditionalReSTIR)
        {
            bool changed = widget.var("Num Integration Prefixes", mpConditionalReSTIRPass->getOptions().subpathSetting.numIntegrationPrefixes, 1, 128);
            bool needReallocate = widget.var("Final Gather Suffixes", mpConditionalReSTIRPass->getOptions().subpathSetting.finalGatherSuffixCount, 1, 8);

            if (changed || needReallocate)
            {
                dirty = true;
                mpConditionalReSTIRPass->mResetTemporalReservoirs = true;
                if (needReallocate)
                {
                    mpConditionalReSTIRPass->mReallocate = true;
                    mpConditionalReSTIRPass->mRecompile = true;
                }
            }
        }
    }

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Stats and debug options.
    renderStatsUI(widget);
    dirty |= renderDebugUI(widget);

    if (dirty)
    {
        validateOptions();
        mOptionsChanged = true;
    }
}

bool PathTracer::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    if (auto group = widget.group("Path Tracer Options", false))
    {
        if (mFixedSampleCount)
        {
            dirty |= widget.var("Samples/pixel", mParams.samplesPerPixel, 1, (int)kMaxSamplesPerPixel);
        }
        else widget.text("Samples/pixel: Variable");
        widget.tooltip("Number of samples per pixel. One path is traced for each sample.\n\n"
            "When the '" + kInputSampleCount + "' input is connected, the number of samples per pixel is loaded from the texture.");

        if (widget.var("Max surface bounces", mStaticParams.maxSurfaceBounces, 0u, kMaxBounces))
        {
            // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
            mStaticParams.maxDiffuseBounces = std::min(mStaticParams.maxDiffuseBounces, mStaticParams.maxSurfaceBounces);
            mStaticParams.maxSpecularBounces = std::min(mStaticParams.maxSpecularBounces, mStaticParams.maxSurfaceBounces);
            mStaticParams.maxTransmissionBounces = std::min(mStaticParams.maxTransmissionBounces, mStaticParams.maxSurfaceBounces);
            dirty = true;
        }
        widget.tooltip("Maximum number of surface bounces (diffuse + specular + transmission).\n"
            "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold are also classified as diffuse events.");

        dirty |= widget.var("Max diffuse bounces", mStaticParams.maxDiffuseBounces, 0u, kMaxBounces);
        widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

        dirty |= widget.var("Max specular bounces", mStaticParams.maxSpecularBounces, 0u, kMaxBounces);
        widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

        dirty |= widget.var("Max transmission bounces", mStaticParams.maxTransmissionBounces, 0u, kMaxBounces);
        widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

        // Sampling options.

        if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
        {
            mpSampleGenerator = SampleGenerator::create(mStaticParams.sampleGenerator);
            dirty = true;
        }

        dirty |= widget.checkbox("BSDF importance sampling", mStaticParams.useBSDFSampling);
        widget.tooltip("BSDF importance sampling should normally be enabled.\n\n"
            "If disabled, cosine-weighted hemisphere sampling is used for debugging purposes");

        dirty |= widget.checkbox("Russian roulette", mStaticParams.useRussianRoulette);
        widget.tooltip("Use russian roulette to terminate low throughput paths.");

        dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
        widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");

        if (mStaticParams.useNEE)
        {
            dirty |= widget.checkbox("Multiple importance sampling (MIS)", mStaticParams.useMIS);
            widget.tooltip("When enabled, BSDF sampling is combined with light sampling for the environment map and emissive lights.\n"
                "Note that MIS has currently no effect on analytic lights.");

            if (mStaticParams.useMIS)
            {
                dirty |= widget.dropdown("MIS heuristic", kMISHeuristicList, reinterpret_cast<uint32_t&>(mStaticParams.misHeuristic));

                if (mStaticParams.misHeuristic == MISHeuristic::PowerExp)
                {
                    dirty |= widget.var("MIS power exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
                }
            }

            if (mpScene && mpScene->useEmissiveLights())
            {
                if (auto group = widget.group("Emissive sampler"))
                {
                    if (widget.dropdown("Emissive sampler", kEmissiveSamplerList, (uint32_t&)mStaticParams.emissiveSampler))
                    {
                        resetLighting();
                        dirty = true;
                    }
                    widget.tooltip("Selects which light sampler to use for importance sampling of emissive geometry.", true);

                    if (mpEmissiveSampler)
                    {
                        if (mpEmissiveSampler->renderUI(group)) mOptionsChanged = true;
                    }
                }
            }
        }
    }

    if (auto group = widget.group("RTXDI"))
    {
        dirty |= widget.checkbox("Enabled", mStaticParams.useRTXDI);
        widget.tooltip("Use RTXDI for direct illumination.");
        if (mpRTXDI) dirty |= mpRTXDI->renderUI(group);
    }

    if (auto group = widget.group("Conditional ReSTIR"))
    {
        dirty |= widget.checkbox("Enabled", mParams.useConditionalReSTIR);
        widget.tooltip("Use Conditional ReSTIR (Final Gather version of ReSTIR PT) for indirect illumination.");
        if (mpConditionalReSTIRPass) dirty |= mpConditionalReSTIRPass->renderUI(group);
    }

    if (auto group = widget.group("Material controls"))
    {
        dirty |= widget.checkbox("Alpha test", mStaticParams.useAlphaTest);
        widget.tooltip("Use alpha testing on non-opaque triangles.");

        dirty |= widget.checkbox("Adjust shading normals on secondary hits", mStaticParams.adjustShadingNormals);
        widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.\nDoes not apply to primary hits which is configured in GBuffer.", true);

        dirty |= widget.var("Max nested materials", mStaticParams.maxNestedMaterials, 2u, 4u);
        widget.tooltip("Maximum supported number of nested materials.");

        dirty |= widget.checkbox("Use lights in dielectric volumes", mStaticParams.useLightsInDielectricVolumes);
        widget.tooltip("Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.");

        dirty |= widget.checkbox("Disable caustics", mStaticParams.disableCaustics);
        widget.tooltip("Disable sampling of caustic light paths (i.e. specular events after diffuse events).");

        runtimeDirty |= widget.var("Specular roughness threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        widget.tooltip("Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.");

        dirty |= widget.dropdown("Primary LOD Mode", kLODModeList, reinterpret_cast<uint32_t&>(mStaticParams.primaryLodMode));
        widget.tooltip("Texture LOD mode at primary hit");

        runtimeDirty |= widget.var("TexLOD bias", mParams.lodBias, -16.f, 16.f, 0.01f);

        dirty |= widget.checkbox("Use Lambertian Diffuse", mStaticParams.useLambertianDiffuse);
        widget.tooltip("Use the simpler Lambertian model for diffuse reflection");

        dirty |= widget.dropdown("DI Mode", kDIMode, reinterpret_cast<uint32_t&>(mParams.DIMode));

        dirty |= widget.checkbox("Disable Diffuse", mStaticParams.disableDiffuse);

        dirty |= widget.checkbox("Disable Specular", mStaticParams.disableSpecular);

        dirty |= widget.checkbox("Disable Translucency", mStaticParams.disableTranslucency);
    }

    if (auto group = widget.group("Denoiser options"))
    {
        dirty |= widget.checkbox("Use NRD demodulation", mStaticParams.useNRDDemodulation);
        widget.tooltip("Global switch for NRD demodulation");
    }

    if (auto group = widget.group("Output options"))
    {
        // Switch to enable/disable path tracer output.
        dirty |= widget.checkbox("Enable output", mEnabled);

        // Controls for output size.
        // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
        if (widget.dropdown("Output size", RenderPassHelpers::kIOSizeList, (uint32_t&)mOutputSizeSelection)) requestRecompile();
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        {
            if (widget.var("Size in pixels", mFixedOutputSize, 32u, 16384u)) requestRecompile();
        }

        dirty |= widget.dropdown("Color format", kColorFormatList, (uint32_t&)mStaticParams.colorFormat);
        widget.tooltip("Selects the color format used for internal per-sample color and denoiser buffers");
    }

    if (dirty) mRecompile = true;
    return dirty || runtimeDirty;
}

bool PathTracer::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging"))
    {
        dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        group.tooltip("Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging.");
        if (mParams.useFixedSeed)
        {
            dirty |= group.var("Seed", mParams.fixedSeed);
        }

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

void PathTracer::renderStatsUI(Gui::Widgets& widget)
{
    if (auto g = widget.group("Statistics"))
    {
        // Show ray stats
        mpPixelStats->renderUI(g);
    }
}

bool PathTracer::onMouseEvent(const MouseEvent& mouseEvent)
{
    bool dirty = mpPixelDebug->onMouseEvent(mouseEvent);
    if (mpConditionalReSTIRPass) dirty |= mpConditionalReSTIRPass->getPixelDebug()->onMouseEvent(mouseEvent);
    return dirty;
}

void PathTracer::setModeId(int modeId)
{
    mRenderModePresetId = modeId;
    setPresetForMethod(mRenderModePresetId, false);
}

void PathTracer::updateDict(const Dictionary& dict)
{
    parseDictionary(dict);
    if (mpConditionalReSTIRPass)
    {
        mpConditionalReSTIRPass->setOptions(mConditionalReSTIROptions);
        mpConditionalReSTIRPass->mReallocate = true;
        mpConditionalReSTIRPass->mRecompile = true;
        mpConditionalReSTIRPass->mResetTemporalReservoirs = true;
    }
    mRecompile = true;
    mOptionsChanged = true;
}

void PathTracer::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getMaterialSystem()->getTypeConformances();

    // Create compute passes.
    Program::Desc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);
    baseDesc.setShaderModel(kShaderModel);

    if (!mpTracePass)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kTracePassFilename).csEntry("main");
        mpTracePass = ComputePass::create(desc, defines, false);
    }

    if (mOutputNRDAdditionalData && (!mpTraceDeltaReflectionPass || !mpTraceDeltaTransmissionPass))
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kTracePassFilename).csEntry("main");

        Program::DefineList deltaReflectionTraceDefine = defines;
        deltaReflectionTraceDefine.add("DELTA_REFLECTION_PASS");
        mpTraceDeltaReflectionPass = ComputePass::create(desc, deltaReflectionTraceDefine, false);

        Program::DefineList deltaTransmissionTraceDefine = defines;
        deltaTransmissionTraceDefine.add("DELTA_TRANSMISSION_PASS");
        mpTraceDeltaTransmissionPass = ComputePass::create(desc, deltaTransmissionTraceDefine, false);
    }

    if (!mpGeneratePaths)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kGeneratePathsFilename).csEntry("main");
        mpGeneratePaths = ComputePass::create(desc, defines, false);
    }
    if (!mpReflectTypes)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(desc, defines, false);
    }

    // Perform program specialization.
    // Note that we must use set instead of add functions to replace any stale state.
    auto prepareProgram = [&](Program::SharedPtr program)
    {
        program->setDefines(defines);
    };
    prepareProgram(mpTracePass->getProgram());
    prepareProgram(mpGeneratePaths->getProgram());
    prepareProgram(mpResolvePass->getProgram());
    prepareProgram(mpReflectTypes->getProgram());

    // Create program vars for the specialized programs.
    mpTracePass->setVars(nullptr);
    if (mpTraceDeltaReflectionPass && mpTraceDeltaTransmissionPass)
    {
        mpTraceDeltaReflectionPass->setVars(nullptr);
        mpTraceDeltaTransmissionPass->setVars(nullptr);
    }
    mpGeneratePaths->setVars(nullptr);
    mpResolvePass->setVars(nullptr);
    mpReflectTypes->setVars(nullptr);

    mVarsChanged = true;
    mRecompile = false;

    // since ReSTIR shares some macro definition with the host program, we need to update as well
    if (mpConditionalReSTIRPass) mpConditionalReSTIRPass->updatePrograms();
}

void PathTracer::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.

    if (mOutputGuideData || mOutputNRDData) mParams.samplesPerPixel = std::min(mParams.samplesPerPixel, 16); // avoid creating large buffer
    uint32_t spp = mFixedSampleCount ? mParams.samplesPerPixel : kMaxSamplesPerPixel;
    uint32_t tileCount = mParams.screenTiles.x * mParams.screenTiles.y;
    const uint32_t sampleCount = tileCount * kScreenTileDim.x * kScreenTileDim.y * spp;
    const uint32_t screenPixelCount = mParams.frameDim.x * mParams.frameDim.y;
    const uint32_t pathCount = screenPixelCount * spp;

    // Allocate output sample offset buffer if needed.
    // This buffer stores the output offset to where the samples for each pixel are stored consecutively.
    // The offsets are local to the current tile, so 16-bit format is sufficient and reduces bandwidth usage.
    if (!mFixedSampleCount)
    {
        if (!mpSampleOffset || mpSampleOffset->getWidth() != mParams.frameDim.x || mpSampleOffset->getHeight() != mParams.frameDim.y)
        {
            FALCOR_ASSERT(kScreenTileDim.x * kScreenTileDim.y * kMaxSamplesPerPixel <= (1u << 16));
            mpSampleOffset = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::R16Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
            mVarsChanged = true;
        }
    }

    auto var = mpReflectTypes->getRootVar();

    if (mOutputGuideData && (!mpSampleGuideData || mpSampleGuideData->getElementCount() < sampleCount || mVarsChanged))
    {
        mpSampleGuideData = Buffer::createStructured(var["sampleGuideData"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mVarsChanged = true;
    }

    if (mOutputNRDData && (!mpSampleNRDRadiance || mpSampleNRDRadiance->getElementCount() < sampleCount || mVarsChanged))
    {
        mpSampleNRDRadiance = Buffer::createStructured(var["sampleNRDRadiance"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDHitDist = Buffer::createStructured(var["sampleNRDHitDist"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDPrimaryHitNeeOnDelta = Buffer::createStructured(var["sampleNRDPrimaryHitNeeOnDelta"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDEmission = Buffer::createStructured(var["sampleNRDEmission"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDPrimaryHitEmission = Buffer::createStructured(var["sampleNRDEmission"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDReflectance = Buffer::createStructured(var["sampleNRDReflectance"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mVarsChanged = true;
    }
}

void PathTracer::preparePathTracer(const RenderData& renderData)
{
    // Create path tracer parameter block if needed.
    if (!mpPathTracerBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(reflector);
        FALCOR_ASSERT(mpPathTracerBlock);
        mVarsChanged = true;
    }

    // Bind resources.
    auto var = mpPathTracerBlock->getRootVar();
    setShaderData(var, renderData);

    // set path tracer shader data for ReSTIR
    if (mParams.useConditionalReSTIR && mpConditionalReSTIRPass)
    {
        if (mVarsChanged || !mpConditionalReSTIRPass->getPathTracerBlock())
            mpConditionalReSTIRPass->createPathTracerBlock();

        auto var = mpConditionalReSTIRPass->getPathTracerBlock()->getRootVar();
        setPathTracerDataForConditionalReSTIR(var, renderData);
    }
}

void PathTracer::resetLighting()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void PathTracer::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for material changes and performs any necessary update.
    // For now all we need to do is to trigger a recompile so that the right defines get set.
    // In the future, we might want to do additional material-specific setup here.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::MaterialsChanged))
    {
        mRecompile = true;
    }
}

bool PathTracer::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getLightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount() > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = EmissiveUniformSampler::create(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler = LightBVHSampler::create(pRenderContext, mpScene, mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = EmissivePowerSampler::create(pRenderContext, mpScene);
                break;
            default:
                throw RuntimeError("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext);
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->getProgram()->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void PathTracer::prepareRTXDI(RenderContext* pRenderContext)
{
    if (mStaticParams.useRTXDI)
    {
        if (!mpRTXDI) mpRTXDI = RTXDI::create(mpScene, mRTXDIOptions);
    }
    else
    {
        mpRTXDI = nullptr;
    }
}

void PathTracer::setNRDData(const ShaderVar& var, const RenderData& renderData) const
{
    var["sampleRadiance"] = mpSampleNRDRadiance;
    var["sampleHitDist"] = mpSampleNRDHitDist;
    var["samplePrimaryHitNEEOnDelta"] = mpSampleNRDPrimaryHitNeeOnDelta;
    var["sampleEmission"] = mpSampleNRDEmission;
    var["samplePrimaryHitEmission"] = mpSampleNRDPrimaryHitEmission;
    var["sampleReflectance"] = mpSampleNRDReflectance;
    var["primaryHitEmission"] = renderData.getTexture(kOutputNRDEmission);
    var["primaryHitDiffuseReflectance"] = renderData.getTexture(kOutputNRDDiffuseReflectance);
    var["primaryHitSpecularReflectance"] = renderData.getTexture(kOutputNRDSpecularReflectance);
    var["deltaReflectionReflectance"] = renderData.getTexture(kOutputNRDDeltaReflectionReflectance);
    var["deltaReflectionEmission"] = renderData.getTexture(kOutputNRDDeltaReflectionEmission);
    var["deltaReflectionNormWRoughMaterialID"] = renderData.getTexture(kOutputNRDDeltaReflectionNormWRoughMaterialID);
    var["deltaReflectionPathLength"] = renderData.getTexture(kOutputNRDDeltaReflectionPathLength);
    var["deltaReflectionHitDist"] = renderData.getTexture(kOutputNRDDeltaReflectionHitDist);
    var["deltaTransmissionReflectance"] = renderData.getTexture(kOutputNRDDeltaTransmissionReflectance);
    var["deltaTransmissionEmission"] = renderData.getTexture(kOutputNRDDeltaTransmissionEmission);
    var["deltaTransmissionNormWRoughMaterialID"] = renderData.getTexture(kOutputNRDDeltaTransmissionNormWRoughMaterialID);
    var["deltaTransmissionPathLength"] = renderData.getTexture(kOutputNRDDeltaTransmissionPathLength);
    var["deltaTransmissionPosW"] = renderData.getTexture(kOutputNRDDeltaTransmissionPosW);
}

void PathTracer::setShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (useLightSampling && mpEnvMapSampler) mpEnvMapSampler->setShaderData(var["envMapSampler"]);

        var["sampleOffset"] = mpSampleOffset; // Can be nullptr
        var["sampleGuideData"] = mpSampleGuideData;
    }

    // Bind runtime data.
    setNRDData(var["outputNRD"], renderData);

    Texture::SharedPtr pViewDir;
    if (mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        pViewDir = renderData.getTexture(kInputViewDir);
        if (!pViewDir) logWarning("Depth-of-field requires the '{}' input. Expect incorrect rendering.", kInputViewDir);
    }

    Texture::SharedPtr pSampleCount;
    if (!mFixedSampleCount)
    {
        pSampleCount = renderData.getTexture(kInputSampleCount);
        if (!pSampleCount) throw RuntimeError("PathTracer: Missing sample count input texture");
    }

    var["params"].setBlob(mParams);
    var["vbuffer"] = renderData.getTexture(kInputVBuffer);
    var["viewDir"] = pViewDir; // Can be nullptr
    var["sampleCount"] = pSampleCount; // Can be nullptr
    var["outputColor"] = renderData.getTexture(kOutputColor);

    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->setShaderData(var["emissiveSampler"]);
    }
    if (mParams.useConditionalReSTIR) mpConditionalReSTIRPass->setShaderData(var["restir"]);
}

bool PathTracer::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update the random seed.
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mParams.frameCount + mSeedOffset;

    if (mpConditionalReSTIRPass) mParams.specularRoughnessThreshold = mpConditionalReSTIRPass->getOptions().shiftMappingSettings.specularRoughnessThreshold;

    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels) {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y)) resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("PathTracer I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Prepare RTXDI.
    prepareRTXDI(pRenderContext);
    if (mpRTXDI) mpRTXDI->beginFrame(pRenderContext, mParams.frameDim);

    // Prepare ReSTIR
    if (mParams.useConditionalReSTIR)
    {
        if (!mpConditionalReSTIRPass)
        {
            auto defines = mStaticParams.getDefines(*this); // set owner defines
            mpConditionalReSTIRPass = ConditionalReSTIRPass::create(mpScene, defines, mConditionalReSTIROptions, mpPixelStats);
        }
    }

    if (mpConditionalReSTIRPass)
    {
        mpConditionalReSTIRPass->setPathTracerParams(mParams.useFixedSeed, mParams.fixedSeed,
            mParams.lodBias, mParams.specularRoughnessThreshold, mParams.frameDim, mParams.screenTiles, mParams.frameCount,
            mParams.seed, mParams.samplesPerPixel, mParams.DIMode);
        mpConditionalReSTIRPass->setSharedStaticParams(0, mStaticParams.maxSurfaceBounces, mStaticParams.useNEE);
        mpConditionalReSTIRPass->beginFrame(pRenderContext, mParams.frameDim, mParams.screenTiles, mRecompile);
    }

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    bool gbufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);
    if (gbufferAdjustShadingNormals != mGBufferAdjustShadingNormals)
    {
        mGBufferAdjustShadingNormals = gbufferAdjustShadingNormals;
        mRecompile = true;
    }

    // Check if fixed sample count should be used. When the sample count input is connected we load the count from there instead.
    mFixedSampleCount = renderData[kInputSampleCount] == nullptr;

    // Check if guide data should be generated.
    mOutputGuideData = renderData[kOutputAlbedo] != nullptr || renderData[kOutputSpecularAlbedo] != nullptr
        || renderData[kOutputIndirectAlbedo] != nullptr || renderData[kOutputNormal] != nullptr
        || renderData[kOutputReflectionPosW] != nullptr;

    // Check if NRD data should be generated.
    mOutputNRDData =
        renderData[kOutputNRDDiffuseRadianceHitDist] != nullptr
        || renderData[kOutputNRDSpecularRadianceHitDist] != nullptr
        || renderData[kOutputNRDResidualRadianceHitDist] != nullptr
        || renderData[kOutputNRDEmission] != nullptr
        || renderData[kOutputNRDDiffuseReflectance] != nullptr
        || renderData[kOutputNRDSpecularReflectance] != nullptr;

    // Check if additional NRD data should be generated.
    bool prevOutputNRDAdditionalData = mOutputNRDAdditionalData;
    mOutputNRDAdditionalData =
        renderData[kOutputNRDDeltaReflectionRadianceHitDist] != nullptr
        || renderData[kOutputNRDDeltaTransmissionRadianceHitDist] != nullptr
        || renderData[kOutputNRDDeltaReflectionReflectance] != nullptr
        || renderData[kOutputNRDDeltaReflectionEmission] != nullptr
        || renderData[kOutputNRDDeltaReflectionNormWRoughMaterialID] != nullptr
        || renderData[kOutputNRDDeltaReflectionPathLength] != nullptr
        || renderData[kOutputNRDDeltaReflectionHitDist] != nullptr
        || renderData[kOutputNRDDeltaTransmissionReflectance] != nullptr
        || renderData[kOutputNRDDeltaTransmissionEmission] != nullptr
        || renderData[kOutputNRDDeltaTransmissionNormWRoughMaterialID] != nullptr
        || renderData[kOutputNRDDeltaTransmissionPathLength] != nullptr
        || renderData[kOutputNRDDeltaTransmissionPosW] != nullptr;
    if (mOutputNRDAdditionalData != prevOutputNRDAdditionalData) mRecompile = true;

    // Enable pixel stats if rayCount or pathLength outputs are connected.
    if (renderData[kOutputRayCount] != nullptr || renderData[kOutputPathLength] != nullptr)
    {
        mpPixelStats->setEnabled(true);
    }

    mpPixelStats->beginFrame(pRenderContext, mParams.frameDim);
    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);

    if (!mpSavedOutput)
    {
        int frameDimX = renderData.getTexture(kOutputColor)->getWidth();
        int frameDimY = renderData.getTexture(kOutputColor)->getHeight();
        mpSavedOutput = Texture::create2D(frameDimX, frameDimY, renderData.getTexture(kOutputColor)->getFormat(), 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }

    return true;
}

void PathTracer::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelStats->endFrame(pRenderContext);
    mpPixelDebug->endFrame(pRenderContext);

    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Copy pixel stats to outputs if available.
    copyTexture(renderData.getTexture(kOutputRayCount).get(), mpPixelStats->getRayCountTexture(pRenderContext).get());
    copyTexture(renderData.getTexture(kOutputPathLength).get(), mpPixelStats->getPathLengthTexture().get());

    if (mpRTXDI) mpRTXDI->endFrame(pRenderContext);
    if (mpConditionalReSTIRPass) mpConditionalReSTIRPass->endFrame(pRenderContext);

    mVarsChanged = false;
    if (!mpScene->freeze) mParams.frameCount++;
}

void PathTracer::generatePaths(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE("generatePaths");

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    FALCOR_ASSERT(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see comments in shader.
    FALCOR_ASSERT(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    FALCOR_ASSERT(mpGeneratePaths->getThreadGroupSize().x == tileSize);
    FALCOR_ASSERT(mpGeneratePaths->getThreadGroupSize().y == 1 && mpGeneratePaths->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    mpGeneratePaths->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    mpGeneratePaths->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    mpGeneratePaths->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");
    mpGeneratePaths->addDefine("OUTPUT_NRD_ADDITIONAL_DATA", mOutputNRDAdditionalData ? "1" : "0");

    // Bind resources.
    auto var = mpGeneratePaths->getRootVar()["CB"]["gPathGenerator"];
    setShaderData(var, renderData, false);
    var["resetTemporal"] = mpConditionalReSTIRPass && mpConditionalReSTIRPass->needResetTemporalHistory();

    mpGeneratePaths["gScene"] = mpScene->getParameterBlock();

    if (mpRTXDI) mpRTXDI->setShaderData(mpGeneratePaths->getRootVar());

    // Launch one thread per pixel.
    // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
    mpGeneratePaths->execute(pRenderContext, { mParams.screenTiles.x * tileSize, mParams.screenTiles.y, 1u });
}

void PathTracer::tracePass(RenderContext* pRenderContext, const RenderData& renderData, const ComputePass::SharedPtr& pTracePass, uint32_t curIter)
{
    FALCOR_PROFILE("Trace Pass");

    // Additional specialization. This shouldn't change resource declarations.
    pTracePass->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    pTracePass->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    pTracePass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");
    pTracePass->addDefine("OUTPUT_NRD_ADDITIONAL_DATA", mOutputNRDAdditionalData ? "1" : "0");

    // Bind global resources.
    auto var = pTracePass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    if (mVarsChanged) mpSampleGenerator->setShaderData(var);
    if (mpRTXDI) mpRTXDI->setShaderData(var);

    mpPixelStats->prepareProgram(pTracePass->getProgram(), var);
    mpPixelDebug->prepareProgram(pTracePass->getProgram(), var);

    // Bind the path tracer.
    var["gPathTracer"] = mpPathTracerBlock;
    var["gScheduler"]["curIter"] = curIter;

    // Full screen dispatch.
    pTracePass->execute(pRenderContext, uint3(mParams.frameDim, 1u));
}

void PathTracer::resolvePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mOutputGuideData && !mOutputNRDData && !mParams.useConditionalReSTIR && mFixedSampleCount && mParams.samplesPerPixel == 1) return;

    FALCOR_PROFILE("resolvePass");

    // This pass is executed when multiple samples per pixel are used.
    // We launch one thread per pixel that computes the resolved color by iterating over the samples.
    // The samples are arranged in tiles with pixels in Morton order, with samples stored consecutively for each pixel.
    // With adaptive sampling, an extra sample offset lookup table computed by the path generation pass is used to
    // locate the samples for each pixel.

    // Additional specialization. This shouldn't change resource declarations.
    mpResolvePass->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    mpResolvePass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");

    // Bind resources.
    auto var = mpResolvePass->getRootVar()["CB"]["gResolvePass"];
    var["params"].setBlob(mParams);
    var["sampleCount"] = renderData.getTexture(kInputSampleCount); // Can be nullptr
    var["outputColor"] = renderData.getTexture(kOutputColor);
    var["outputAlbedo"] = renderData.getTexture(kOutputAlbedo);
    var["outputSpecularAlbedo"] = renderData.getTexture(kOutputSpecularAlbedo);
    var["outputIndirectAlbedo"] = renderData.getTexture(kOutputIndirectAlbedo);
    var["outputNormal"] = renderData.getTexture(kOutputNormal);
    var["outputReflectionPosW"] = renderData.getTexture(kOutputReflectionPosW);
    var["outputNRDDiffuseRadianceHitDist"] = renderData.getTexture(kOutputNRDDiffuseRadianceHitDist);
    var["outputNRDSpecularRadianceHitDist"] = renderData.getTexture(kOutputNRDSpecularRadianceHitDist);
    var["outputNRDDeltaReflectionRadianceHitDist"] = renderData.getTexture(kOutputNRDDeltaReflectionRadianceHitDist);
    var["outputNRDDeltaTransmissionRadianceHitDist"] = renderData.getTexture(kOutputNRDDeltaTransmissionRadianceHitDist);
    var["outputNRDResidualRadianceHitDist"] = renderData.getTexture(kOutputNRDResidualRadianceHitDist);
    var["vbuffer"] = renderData.getTexture(kInputVBuffer);
    if (mpConditionalReSTIRPass)
    {
        mpConditionalReSTIRPass->setReservoirData(var);
    }

    if (mVarsChanged)
    {
        var["sampleOffset"] = mpSampleOffset; // Can be nullptr
        var["sampleGuideData"] = mpSampleGuideData;
        var["sampleNRDRadiance"] = mpSampleNRDRadiance;
        var["sampleNRDHitDist"] = mpSampleNRDHitDist;
        var["sampleNRDEmission"] = mpSampleNRDEmission;
        var["sampleNRDReflectance"] = mpSampleNRDReflectance;

        var["sampleNRDPrimaryHitNeeOnDelta"] = mpSampleNRDPrimaryHitNeeOnDelta;
        var["primaryHitDiffuseReflectance"] = renderData.getTexture(kOutputNRDDiffuseReflectance);
    }
    var["outputColorPrev"] = mpSavedOutput;
    var["freeze"] = mpScene->freeze && mIsFrozen;

    // Launch one thread per pixel.
    mpResolvePass->execute(pRenderContext, { mParams.frameDim, 1u });
}

void PathTracer::setPathTracerDataForConditionalReSTIR(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (useLightSampling && mpEnvMapSampler) mpEnvMapSampler->setShaderData(var["envMapSampler"]);
    }

    Texture::SharedPtr pViewDir;
    if (mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        pViewDir = renderData.getTexture(kInputViewDir);
        if (!pViewDir) logWarning("Depth-of-field requires the '{}' input. Expect incorrect rendering.", kInputViewDir);
    }

    Texture::SharedPtr pSampleCount;
    if (!mFixedSampleCount)
    {
        pSampleCount = renderData.getTexture(kInputSampleCount);
        if (!pSampleCount) throw RuntimeError("PathTracer: Missing sample count input texture");
    }

    var["params"].setBlob(mParams);
    var["vbuffer"] = renderData.getTexture(kInputVBuffer);
    var["viewDir"] = pViewDir; // Can be nullptr
    var["sampleCount"] = pSampleCount; // Can be nullptr
    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->setShaderData(var["emissiveSampler"]);
    }
    var["outputColor"] = renderData.getTexture(kOutputColor);

    if (mParams.useConditionalReSTIR) mpConditionalReSTIRPass->setShaderData(var["restir"]);
}

void PathTracer::setPresetForMethod(int id, bool fromGui)
{
    mpConditionalReSTIRPass->mReallocate = true;
    mpConditionalReSTIRPass->mRecompile = true;
    mpConditionalReSTIRPass->mResetTemporalReservoirs = true;
    mRecompile = true;
    mOptionsChanged = true;

    mSavedPTSpp[mPrevRenderModePresetId] = mParams.samplesPerPixel;
    mPrevRenderModePresetId = id;

    if (id == 0) // Conditional ReSTIR
    {
        mParams.samplesPerPixel = 1;
        mpConditionalReSTIRPass->getOptions().subpathSetting.useMMIS = false;
        mpConditionalReSTIRPass->getOptions().shiftMapping = ConditionalReSTIR::ShiftMapping::Hybrid;
        mpConditionalReSTIRPass->getOptions().subpathSetting.suffixSpatialReuseRounds = 1;
        mpConditionalReSTIRPass->getOptions().subpathSetting.temporalHistoryLength = 50;
        mpConditionalReSTIRPass->getOptions().subpathSetting.suffixTemporalReuse = true;
        mParams.useConditionalReSTIR = true;
    } 
    else if (id == 1) // MMIS
    {
        mParams.samplesPerPixel = 1;
        mpConditionalReSTIRPass->getOptions().subpathSetting.useMMIS = true;
        mpConditionalReSTIRPass->getOptions().shiftMapping = ConditionalReSTIR::ShiftMapping::Reconnection;
        mpConditionalReSTIRPass->getOptions().subpathSetting.suffixSpatialReuseRounds = 0;
        mpConditionalReSTIRPass->getOptions().subpathSetting.temporalHistoryLength = 0;
        mpConditionalReSTIRPass->getOptions().subpathSetting.suffixTemporalReuse = false;
        mParams.useConditionalReSTIR = true;
    }
    else if (id == 2) // path tracing
    {
        if (fromGui) mParams.samplesPerPixel = mSavedPTSpp[2];
        mpConditionalReSTIRPass->getOptions().subpathSetting.useMMIS = false;
        mpConditionalReSTIRPass->getOptions().shiftMapping = ConditionalReSTIR::ShiftMapping::Hybrid;
        mpConditionalReSTIRPass->getOptions().subpathSetting.suffixSpatialReuseRounds = 1;
        mpConditionalReSTIRPass->getOptions().subpathSetting.temporalHistoryLength = 50;
        mpConditionalReSTIRPass->getOptions().subpathSetting.suffixTemporalReuse = true;
        mParams.useConditionalReSTIR = false;
    }

}



Program::DefineList PathTracer::StaticParams::getDefines(const PathTracer& owner) const
{
    Program::DefineList defines;

    // Path tracer configuration.
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(maxSurfaceBounces));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(maxTransmissionBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS && useNEE ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_RTXDI", useRTXDI ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("USE_NRD_DEMODULATION", useNRDDemodulation ? "1" : "0");
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());
    if (owner.mpRTXDI) defines.add(owner.mpRTXDI->getDefines());

    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene) defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_CURVES", scene && (scene->hasGeometryType(Scene::GeometryType::Curve)) ? "1" : "0");
    defines.add("USE_SDF_GRIDS", scene && scene->hasGeometryType(Scene::GeometryType::SDFGrid) ? "1" : "0");
    defines.add("USE_HAIR_MATERIAL", scene && scene->getMaterialCountByType(MaterialType::Hair) > 0u ? "1" : "0");

    // Set default (off) values for additional features.
    defines.add("USE_VIEW_DIR", "0");
    defines.add("OUTPUT_GUIDE_DATA", "0");
    defines.add("OUTPUT_NRD_DATA", "0");
    defines.add("OUTPUT_NRD_ADDITIONAL_DATA", "0");

    defines.add("DiffuseBrdf", useLambertianDiffuse ? "DiffuseBrdfLambert" : "DiffuseBrdfFrostbite");
    defines.add("enableDiffuse", disableDiffuse ? "0" : "1");
    defines.add("enableSpecular", disableSpecular ? "0" : "1");
    defines.add("enableTranslucency", disableTranslucency ? "0" : "1");

    if (owner.mpConditionalReSTIRPass)
    {
        owner.mpConditionalReSTIRPass->setOwnerDefines(defines);
        defines.add(owner.mpConditionalReSTIRPass->getDefines());
    }

    return defines;
}
