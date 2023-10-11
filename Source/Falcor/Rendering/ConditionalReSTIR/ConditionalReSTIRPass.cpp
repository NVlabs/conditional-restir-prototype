/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#include "ConditionalReSTIRPass.h"
#include "Core/Assert.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Color/ColorHelpers.slang"
#include "../../../RenderPasses/PathTracer/Params.slang"

namespace Falcor
{
    namespace
    {
        const char kReflectTypesFile[] = "Rendering/ConditionalReSTIR/ReflectTypes.cs.slang";
        const char kSuffixSpatialResamplingFile[] = "Rendering/ConditionalReSTIR/SuffixResampling.cs.slang";
        const char kSuffixTemporalResamplingFile[] = "Rendering/ConditionalReSTIR/SuffixResampling.cs.slang";
        const char kSuffixResamplingFile[] = "Rendering/ConditionalReSTIR/SuffixResampling.cs.slang";
        const char kTemporalResamplingFile[] = "Rendering/ConditionalReSTIR/TemporalResampling.cs.slang";
        const char kSpatialResamplingFile[] = "Rendering/ConditionalReSTIR/SpatialResampling.cs.slang";
        const char kTemporalRetraceFile[] = "Rendering/ConditionalReSTIR/TemporalPathRetrace.cs.slang";
        const char kSpatialRetraceFile[] = "Rendering/ConditionalReSTIR/SpatialPathRetrace.cs.slang";
        const char kProduceRetraceWorkload[] = "Rendering/ConditionalReSTIR/ProduceRetraceWorkload.cs.slang";
        const char kSuffixRetraceFile[] = "Rendering/ConditionalReSTIR/SuffixPathRetrace.cs.slang";
        const char kSuffixProduceRetraceWorkload[] = "Rendering/ConditionalReSTIR/SuffixProduceRetraceWorkload.cs.slang";
        const char kSuffixRetraceTalbotFile[] = "Rendering/ConditionalReSTIR/SuffixPathRetraceTalbot.cs.slang";
        const char kSuffixProduceRetraceTalbotWorkload[] = "Rendering/ConditionalReSTIR/SuffixProduceRetraceTalbotWorkload.cs.slang";

        const char kPrefixRetraceFile[] = "Rendering/ConditionalReSTIR/PrefixPathRetrace.cs.slang";
        const char kPrefixProduceRetraceWorkload[] = "Rendering/ConditionalReSTIR/PrefixProduceRetraceWorkload.cs.slang";
        const char kPrefixResampling[] = "Rendering/ConditionalReSTIR/PrefixResampling.cs.slang";

        const char kTraceNewSuffixes[] = "Rendering/ConditionalReSTIR/TraceNewSuffixes.cs.slang";
        const char kPrefixNeighborSearch[] = "Rendering/ConditionalReSTIR/PrefixNeighborSearch.cs.slang";
        const char kTraceNewPrefixes[] = "Rendering/ConditionalReSTIR/TraceNewPrefixes.cs.slang";

        const std::string kShaderModel = "6_5";

        const Gui::DropdownList kShiftMappingList =
        {
            { (uint32_t)ConditionalReSTIR::ShiftMapping::Reconnection, "Reconnection" },
            { (uint32_t)ConditionalReSTIR::ShiftMapping::Hybrid, "Hybrid" },
        };

        const Gui::DropdownList kRetraceScheduleType =
        {
            { (uint32_t)ConditionalReSTIR::RetraceScheduleType::Naive, "Naive" },
            { (uint32_t)ConditionalReSTIR::RetraceScheduleType::Compact, "Compact" },
        };

        const Gui::DropdownList kKNNAdaptiveRadiusType = {
            {(uint32_t)ConditionalReSTIR::KNNAdaptiveRadiusType::NonAdaptive, "NonAdaptive"},
            {(uint32_t)ConditionalReSTIR::KNNAdaptiveRadiusType::RayCone, "RayCone"},
        };

        const uint32_t kNeighborOffsetCount = 8192;
    }

    ConditionalReSTIRPass::SharedPtr ConditionalReSTIRPass::create(const Scene::SharedPtr& pScene, const Program::DefineList& ownerDefines, const Options& options, const PixelStats::SharedPtr& pPixelStats)
    {
        return SharedPtr(new ConditionalReSTIRPass(pScene, ownerDefines, options, pPixelStats));
    }


    void ConditionalReSTIRPass::createComputePass(ComputePass::SharedPtr& pPass, std::string shaderFile, Program::DefineList defines, Program::Desc baseDesc, std::string entryFunction)
    {
        if (!pPass)
        {
            Program::Desc desc = baseDesc;
            desc.addShaderLibrary(shaderFile).csEntry(entryFunction == "" ? "main" : entryFunction);
            pPass = ComputePass::create(desc, defines, false);
        }
        pPass->getProgram()->addDefines(defines);
        pPass->setVars(nullptr);
    }


    ConditionalReSTIRPass::ConditionalReSTIRPass(const Scene::SharedPtr& pScene, const Program::DefineList& ownerDefines, const Options& options, const PixelStats::SharedPtr& pPixelStats)
        : mpScene(pScene)
        , mOptions(options)
    {
        FALCOR_ASSERT(mpScene);

        mpPixelDebug = PixelDebug::create();

        // Create compute pass for reflecting data types.
        Program::Desc desc;
        Program::DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(ownerDefines);
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main").setShaderModel(kShaderModel);
        mpReflectTypes = ComputePass::create(desc, defines);

        // Create neighbor offset texture.
        mpNeighborOffsets = createNeighborOffsetTexture(kNeighborOffsetCount);

        mpPixelStats = pPixelStats;
    }

    Program::DefineList ConditionalReSTIRPass::getDefines() const
    {
        Program::DefineList defines;
        defines.add("TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE", mOptions.temporalUpdateForDynamicScene ? "1": "0");
        defines.add("USE_RESERVOIR_COMPRESSION", mOptions.useReservoirCompression ? "1" : "0");
        defines.add("RETRACE_SCHEDULE_TYPE", std::to_string((uint32_t)mOptions.retraceScheduleType));
        defines.add("COMPRESS_PREFIX_SEARCH_ENTRY", mOptions.subpathSetting.compressNeighborSearchKey ? "1" : "0");
        defines.add("USE_PREV_FRAME_SCENE_DATA", mOptions.usePrevFrameSceneData ? "1" : "0");

        return defines;
    }

    void ConditionalReSTIRPass::setShaderData(const ShaderVar& var) const
    {
        var["settings"]["localStrategyType"] = mOptions.shiftMappingSettings.localStrategyType;
        var["settings"]["specularRoughnessThreshold"] = mOptions.shiftMappingSettings.specularRoughnessThreshold;
        var["settings"]["nearFieldDistanceThreshold"] = mOptions.shiftMappingSettings.nearFieldDistanceThreshold;

        var["subpathSettings"]["adaptivePrefixLength"] = mOptions.subpathSetting.adaptivePrefixLength;
        var["subpathSettings"]["avoidSpecularPrefixEndVertex"] = mOptions.subpathSetting.avoidSpecularPrefixEndVertex;
        var["subpathSettings"]["avoidShortPrefixEndSegment"] = mOptions.subpathSetting.avoidShortPrefixEndSegment;
        var["subpathSettings"]["shortSegmentThreshold"] = mOptions.subpathSetting.shortSegmentThreshold;

        var["subpathSettings"]["suffixSpatialNeighborCount"] = mOptions.subpathSetting.suffixSpatialNeighborCount;
        var["subpathSettings"]["suffixSpatialReuseRadius"] = mOptions.subpathSetting.suffixSpatialReuseRadius;
        var["subpathSettings"]["suffixSpatialReuseRounds"] = mOptions.subpathSetting.suffixSpatialReuseRounds;
        var["subpathSettings"]["numIntegrationPrefixes"] = mOptions.subpathSetting.numIntegrationPrefixes;
        var["subpathSettings"]["generateCanonicalSuffixForEachPrefix"] = mOptions.subpathSetting.generateCanonicalSuffixForEachPrefix;

        var["subpathSettings"]["suffixTemporalReuse"] = mOptions.subpathSetting.suffixTemporalReuse;
        var["subpathSettings"]["temporalHistoryLength"] = mOptions.subpathSetting.temporalHistoryLength;

        var["subpathSettings"]["prefixNeighborSearchRadius"] = mOptions.subpathSetting.prefixNeighborSearchRadius;
        var["subpathSettings"]["prefixNeighborSearchNeighborCount"] = mOptions.subpathSetting.prefixNeighborSearchNeighborCount;
        var["subpathSettings"]["finalGatherSuffixCount"] = mOptions.subpathSetting.finalGatherSuffixCount;

        var["subpathSettings"]["useTalbotMISForGather"] = mOptions.subpathSetting.useTalbotMISForGather;
        var["subpathSettings"]["nonCanonicalWeightMultiplier"] = mOptions.subpathSetting.nonCanonicalWeightMultiplier;
        var["subpathSettings"]["disableCanonical"] = mOptions.subpathSetting.disableCanonical;
        var["subpathSettings"]["compressNeighborSearchKey"] = mOptions.subpathSetting.compressNeighborSearchKey;


        var["subpathSettings"]["knnSearchRadiusMultiplier"] = mOptions.subpathSetting.knnSearchRadiusMultiplier;
        var["subpathSettings"]["knnSearchAdaptiveRadiusType"] = mOptions.subpathSetting.knnSearchAdaptiveRadiusType;
        var["subpathSettings"]["knnIncludeDirectionSearch"] = mOptions.subpathSetting.knnIncludeDirectionSearch;

        var["subpathSettings"]["useMMIS"] = mOptions.subpathSetting.useMMIS;

        var["minimumPrefixLength"] = mOptions.minimumPrefixLength;

        int numRounds = mOptions.subpathSetting.suffixSpatialReuseRounds + 1; //include the prefix streaming pass
        numRounds = mOptions.subpathSetting.suffixTemporalReuse ? numRounds + 1 : numRounds;
        var["suffixSpatialRounds"] = numRounds;
        var["pathReservoirs"] = mpScratchReservoirs;
        var["prefixGBuffer"] = mpScratchPrefixGBuffer;
        var["prefixPathReservoirs"] = mpPrefixPathReservoirs;
        var["prefixThroughputs"] = mpPrefixThroughputs;
        var["prefixReservoirs"] = mpPrefixReservoirs;
        float3 worldBoundExtent = mpScene->getSceneBounds().extent();
        var["sceneRadius"] = std::min(worldBoundExtent.x, std::min(worldBoundExtent.y, worldBoundExtent.z));
        
        var["needResetTemporalHistory"] = mResetTemporalReservoirs;
        var["samplesPerPixel"] = mPathTracerParams.samplesPerPixel;
        var["shiftMapping"] = (uint32_t)mOptions.shiftMapping;
    }

    void ConditionalReSTIRPass::setPathTracerParams(int useFixedSeed, uint fixedSeed,
    float lodBias, float specularRoughnessThreshold, uint2 frameDim, uint2 screenTiles, uint frameCount, uint seed,
    int samplesPerPixel, int DIMode)
    {
        mPathTracerParams.useFixedSeed = useFixedSeed;
        mPathTracerParams.fixedSeed = fixedSeed;
        mPathTracerParams.lodBias = lodBias;
        mPathTracerParams.specularRoughnessThreshold = specularRoughnessThreshold;
        mPathTracerParams.frameDim = frameDim;
        mPathTracerParams.screenTiles = screenTiles;
        mPathTracerParams.frameCount = frameCount;
        mPathTracerParams.seed = seed;
        mPathTracerParams.samplesPerPixel = samplesPerPixel;
        mPathTracerParams.DIMode = DIMode;
    }

    void ConditionalReSTIRPass::setOwnerDefines(Program::DefineList defines)
    {
        mOwnerDefines = defines;
    }

    void ConditionalReSTIRPass::setSharedStaticParams(uint32_t samplesPerPixel, uint32_t maxSurfaceBounces, bool useNEE)
    {
        mStaticParams.maxSurfaceBounces = maxSurfaceBounces;
        mStaticParams.useNEE = useNEE;
    }

    void ConditionalReSTIRPass::createPathTracerBlock()
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(reflector);
    }

    ParameterBlock::SharedPtr ConditionalReSTIRPass::getPathTracerBlock()
    {
        return mpPathTracerBlock;
    }

    void ConditionalReSTIRPass::setReservoirData(const ShaderVar& var) const
    {
        var["pathReservoirs"] = mpReservoirs;
    }

    bool ConditionalReSTIRPass::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        if (auto group = widget.group("Performance settings", true))
        {
            mReallocate |= group.checkbox("Use reservoir compression", mOptions.useReservoirCompression);
            mReallocate |= group.dropdown("Retrace Schedule Type", kRetraceScheduleType, reinterpret_cast<uint32_t&>(mOptions.retraceScheduleType));
        }

        if (auto group = widget.group("Subpath reuse", true))
        {
            dirty |= group.var("Num Integration Prefixes", mOptions.subpathSetting.numIntegrationPrefixes, 1, 128);
            dirty |= group.checkbox("Generate Canonical Suffix For Each Prefix", mOptions.subpathSetting.generateCanonicalSuffixForEachPrefix);
            dirty |= group.checkbox("Use MMIS", mOptions.subpathSetting.useMMIS);
            dirty |= group.var("Min Prefix Length", mOptions.minimumPrefixLength, 1u, mStaticParams.maxSurfaceBounces);
            dirty |= group.checkbox("Adaptive Prefix Length", mOptions.subpathSetting.adaptivePrefixLength);
            dirty |= group.checkbox("Avoid Specular Prefix End Vertex", mOptions.subpathSetting.avoidSpecularPrefixEndVertex);
            dirty |= group.checkbox("Avoid Short Prefix End Segment", mOptions.subpathSetting.avoidShortPrefixEndSegment);
            dirty |= group.var("Short Segment Threshold", mOptions.subpathSetting.shortSegmentThreshold, 0.f, 0.1f);

            mReallocate |= group.var("Suffix Spatial Neighbors", mOptions.subpathSetting.suffixSpatialNeighborCount, 1, 8);
            dirty |= group.var("Suffix Spatial Reuse Radius", mOptions.subpathSetting.suffixSpatialReuseRadius, 0.f, 100.f);

            {
                dirty |= group.var("Suffix Reuse rounds", mOptions.subpathSetting.suffixSpatialReuseRounds, 0, 16);
                dirty |= group.checkbox("Suffix Temporal Reuse", mOptions.subpathSetting.suffixTemporalReuse);
                dirty |= group.var("Suffix Temopral History Length", mOptions.subpathSetting.temporalHistoryLength, 0, 100);

                mReallocate |= group.var("Final Gather Suffix Count", mOptions.subpathSetting.finalGatherSuffixCount, 1, 8);
                mReallocate |= group.checkbox("Use Talbot MIS For Gather", mOptions.subpathSetting.useTalbotMISForGather);
                dirty |= group.var("Non-Canonical Weight Multiplier", mOptions.subpathSetting.nonCanonicalWeightMultiplier, 0.f, 100.f);
                dirty |= group.checkbox("Disable Canonical", mOptions.subpathSetting.disableCanonical);

                dirty |= group.var("KNN Search Radius Multiplier", mOptions.subpathSetting.knnSearchRadiusMultiplier);
                dirty |= group.dropdown("KNN Search Adaptive Type", kKNNAdaptiveRadiusType, reinterpret_cast<uint32_t&>(mOptions.subpathSetting.knnSearchAdaptiveRadiusType));
                dirty |= group.checkbox("KNN Inlucde Direction Search For Low Roughness", mOptions.subpathSetting.knnIncludeDirectionSearch);
                if (mOptions.subpathSetting.knnIncludeDirectionSearch)
                {
                    dirty |= group.var("Final Gather Screen Search Radius", mOptions.subpathSetting.prefixNeighborSearchRadius, 0, 100);
                    dirty |= group.var("Final Gather Screen Search Neighbors", mOptions.subpathSetting.prefixNeighborSearchNeighborCount, 0, 100);
                }

                mReallocate |= group.checkbox("Compress Neighbor Search Key", mOptions.subpathSetting.compressNeighborSearchKey);
            }
        }

        if (auto group = widget.group("Shift mapping options", true))
        {
            mRecompile |= group.dropdown("Shift Mapping", kShiftMappingList, reinterpret_cast<uint32_t&>(mOptions.shiftMapping));

            if (mOptions.shiftMapping == ConditionalReSTIR::ShiftMapping::Hybrid)
            {
                dirty |= group.var("Distance Threshold", mOptions.shiftMappingSettings.nearFieldDistanceThreshold);
                dirty |= group.var("Roughness Threshold", mOptions.shiftMappingSettings.specularRoughnessThreshold);
            }
        }

        mReallocate |= widget.checkbox("Temporal Reservoir Update for Dynamic Scenes", mOptions.temporalUpdateForDynamicScene);

        mRecompile |= widget.checkbox("Use Prev Frame Scene Data", mOptions.usePrevFrameSceneData);


        if (auto group = widget.group("Debugging"))
        {
            mpPixelDebug->renderUI(group);
        }

        mRecompile |= mReallocate;
        dirty |= mRecompile;

        if (dirty) mResetTemporalReservoirs = true;

        return dirty;
    }

    void ConditionalReSTIRPass::setOptions(const Options& options)
    {
        if (std::memcmp(&options, &mOptions, sizeof(Options)) != 0)
        {
            mOptions = options;
            mRecompile = true;
        }
    }

    void ConditionalReSTIRPass::beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile)
    {
        mRecompile |= needRecompile;

        mFrameDim = frameDim;

        prepareResources(pRenderContext, frameDim, screenTiles);

        mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
    }

    void ConditionalReSTIRPass::endFrame(RenderContext* pRenderContext)
    {
        mFrameIndex++;

        // Swap reservoirs.
        if (!mpScene->freeze)
        {
            std::swap(mpPrefixReservoirs, mpPrevPrefixReservoirs);
            std::swap(mpReservoirs, mpPrevReservoirs);
            std::swap(mpPrefixGBuffer, mpPrevPrefixGBuffer);
        }

        mpPixelDebug->endFrame(pRenderContext);
    }

    void ConditionalReSTIRPass::createOrDestroyBuffer(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = Buffer::createStructured(mpReflectTypes[reflectVarName], requiredElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ConditionalReSTIRPass::createOrDestroyBufferWithCounter(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = Buffer::createStructured(mpReflectTypes[reflectVarName], requiredElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, true);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ConditionalReSTIRPass::createOrDestroyBufferNoReallocate(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (!pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = Buffer::createStructured(mpReflectTypes[reflectVarName], requiredElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ConditionalReSTIRPass::createOrDestroyBufferWithCounterNoReallocate(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (!pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = Buffer::createStructured(mpReflectTypes[reflectVarName], requiredElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, true);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ConditionalReSTIRPass::createOrDestroyRawBuffer(Buffer::SharedPtr& pBuffer, size_t requiredSize, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getSize() != requiredSize))
            pBuffer = Buffer::create(requiredSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ConditionalReSTIRPass::prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles)
    {
        // disable hybrid shift, temporal
        if (mReallocate && mpReservoirs) updatePrograms();
        // Create screen sized buffers.
        uint32_t tileCount = screenTiles.x * screenTiles.y;
        const uint32_t elementCount = tileCount * kScreenTileDim.x * kScreenTileDim.y;

        // getting correct struct sizes when initializing
        if (!mpReservoirs)
        {
            Program::DefineList defines;
            defines.add(getDefines());
            Program::TypeConformanceList typeConformances;
            // Scene-specific configuration.
            typeConformances.add(mpScene->getTypeConformances());
            Program::Desc baseDesc;
            baseDesc.addShaderModules(mpScene->getShaderModules());
            baseDesc.addTypeConformances(typeConformances);
            baseDesc.setShaderModel(kShaderModel);
            createComputePass(mpReflectTypes, kReflectTypesFile, defines, baseDesc);
        }

        createOrDestroyBuffer(mpReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrevReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpScratchReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrefixPathReservoirs, "prefixPathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrefixThroughputs, "prefixThroughputs", elementCount);

        createOrDestroyBuffer(mpPrevSuffixReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpTempReservoirs, "pathReservoirs", elementCount, mpScene->freeze);
        createOrDestroyBuffer(mpNeighborValidMaskBuffer, "neighborValidMask", elementCount);

        // for hybrid shift workload compaction
        int maxNeighborCount = std::max( mOptions.subpathSetting.finalGatherSuffixCount, mOptions.subpathSetting.suffixSpatialNeighborCount);
        const uint32_t talbotPathCount = elementCount * (mOptions.subpathSetting.useTalbotMISForGather ? mOptions.subpathSetting.finalGatherSuffixCount * (mOptions.subpathSetting.finalGatherSuffixCount + 1) : 0);
        const uint32_t pathCount = std::max(talbotPathCount, elementCount * 2 * maxNeighborCount);

        createOrDestroyRawBuffer(mpWorkload, pathCount * sizeof(uint32_t), mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact);
        createOrDestroyRawBuffer(mpWorkloadExtra, pathCount * sizeof(uint32_t), mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather);

        createOrDestroyRawBuffer(mpCounter, sizeof(uint32_t), mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact);

        //4*1024*1024*1024 / 48 (rcReconnectionData size at compress reservoir and sd_optim), and round to nearest 10^7 (somehow using originla number causes crash)
        createOrDestroyBuffer(mpReconnectionDataBuffer, "reconnectionDataBuffer", std::min(80000000u, pathCount));
        createOrDestroyBuffer(mpRcBufferOffsets, "rcBufferOffsets", pathCount);

        const uint32_t maxVertexCount = elementCount * (mStaticParams.maxSurfaceBounces + 1);

        createOrDestroyBuffer(mpPrefixGBuffer, "prefixGBuffer", elementCount);
        createOrDestroyBuffer(mpPrevPrefixGBuffer, "prefixGBuffer", elementCount);
        createOrDestroyBuffer(mpFinalGatherSearchKeys, "prefixSearchKeys", elementCount);

        createOrDestroyBuffer(mpPrefixReservoirs, "prefixReservoirs", elementCount);
        createOrDestroyBuffer(mpPrevPrefixReservoirs, "prefixReservoirs", elementCount);

        createOrDestroyBuffer(mpScratchPrefixGBuffer, "prefixGBuffer", elementCount);

        createOrDestroyBuffer(mpFoundNeighborPixels, "foundNeighborPixels", mOptions.subpathSetting.finalGatherSuffixCount * elementCount);

        createOrDestroyBufferWithCounterNoReallocate(mpSearchPointBoundingBoxBuffer, "searchPointBoundingBoxBuffer", frameDim.x * frameDim.y);
        createOrDestroyBufferNoReallocate(mpPrefixL2LengthBuffer, "prefixL2LengthBuffer", frameDim.x * frameDim.y);

        if (!mpTemporalVBuffer || mpTemporalVBuffer->getHeight() != frameDim.y || mpTemporalVBuffer->getWidth() != frameDim.x)
        {
            mpTemporalVBuffer = Texture::create2D(frameDim.x, frameDim.y, mpScene->getHitInfo().getFormat(), 1, 1);
        }

        mReallocate = false;
    }

    void ConditionalReSTIRPass::updatePrograms()
    {
         if (!mRecompile) return;

         Program::DefineList commonDefines;

         commonDefines.add(getDefines());
         commonDefines.add(mpScene->getSceneDefines());
         commonDefines.add(mOwnerDefines);

         Program::TypeConformanceList typeConformances;
         // Scene-specific configuration.
         typeConformances.add(mpScene->getTypeConformances());

         Program::Desc baseDesc;
         baseDesc.addShaderModules(mpScene->getShaderModules());
         baseDesc.addTypeConformances(typeConformances);
         baseDesc.setShaderModel(kShaderModel);

         Program::DefineList defines = commonDefines;
         defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(mpNeighborOffsets->getWidth()));

         createComputePass(mpReflectTypes, kReflectTypesFile, defines, baseDesc);
         createComputePass(mpPrefixResampling, kPrefixResampling, defines, baseDesc);
         createComputePass(mpTraceNewSuffixes, kTraceNewSuffixes, defines, baseDesc);
         createComputePass(mpTraceNewPrefixes, kTraceNewPrefixes, defines, baseDesc);
         createComputePass(mpPrefixNeighborSearch, kPrefixNeighborSearch, defines, baseDesc);
         createComputePass(mpSuffixSpatialResampling, kSuffixSpatialResamplingFile, defines, baseDesc, "spatial");
         createComputePass(mpSuffixTemporalResampling, kSuffixTemporalResamplingFile, defines, baseDesc, "temporal");
         createComputePass(mpSuffixResampling, kSuffixResamplingFile, defines, baseDesc, "gather");
         createComputePass(mpPrefixRetrace, kPrefixRetraceFile, defines, baseDesc);
         createComputePass(mpPrefixProduceRetraceWorkload, kPrefixProduceRetraceWorkload, defines, baseDesc);
         createComputePass(mpSuffixRetrace, kSuffixRetraceFile, defines, baseDesc);
         createComputePass(mpSuffixProduceRetraceWorkload, kSuffixProduceRetraceWorkload, defines, baseDesc);
         createComputePass(mpSuffixRetraceTalbot, kSuffixRetraceTalbotFile, defines, baseDesc);
         createComputePass(mpSuffixProduceRetraceTalbotWorkload, kSuffixProduceRetraceTalbotWorkload, defines, baseDesc);

         mRecompile = false;
         mResetTemporalReservoirs = true;
    }

    void ConditionalReSTIRPass::suffixResamplingPass(
        RenderContext* pRenderContext,
        const Texture::SharedPtr& pVBuffer,
        const Texture::SharedPtr& pMotionVectors,
        const Texture::SharedPtr& pOutputColor
    )
    {
        FALCOR_PROFILE("SuffixResampling");

        bool hasTemporalReuse = mOptions.subpathSetting.suffixTemporalReuse;
        // if we have no temporal history, skip the first round (set suffixTemporalReuse in CB to false temporarily)
        mOptions.subpathSetting.suffixTemporalReuse = (mResetTemporalReservoirs) ? false : mOptions.subpathSetting.suffixTemporalReuse;

        ShaderVar presamplingVar = bindSuffixResamplingVars(pRenderContext, mpPrefixResampling, "gPrefixResampling", pVBuffer, pMotionVectors, true, true);
        presamplingVar["prevCameraU"] = mPrevCameraU;
        presamplingVar["prevCameraV"] = mPrevCameraV;
        presamplingVar["prevCameraW"] = mPrevCameraW;
        presamplingVar["prevJitterX"] = mPrevJitterX;
        presamplingVar["prevJitterY"] = mPrevJitterY;
        presamplingVar["prefixReservoirs"] = mpPrefixReservoirs;
        presamplingVar["prevPrefixReservoirs"] = mpPrevPrefixReservoirs;
        presamplingVar["rcBufferOffsets"] = mpRcBufferOffsets;
        presamplingVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;

        ShaderVar spatialVar = bindSuffixResamplingVars(pRenderContext, mpSuffixSpatialResampling, "gSuffixResampling", pVBuffer, pMotionVectors, true, false);
        spatialVar["outColor"] = pOutputColor;
        spatialVar["rcBufferOffsets"] = mpRcBufferOffsets;
        spatialVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        spatialVar["foundNeighborPixels"] = mpFoundNeighborPixels;

        ShaderVar temporalVar = bindSuffixResamplingVars(pRenderContext, mpSuffixTemporalResampling, "gSuffixResampling", pVBuffer, pMotionVectors, true, false);
        temporalVar["outColor"] = pOutputColor;
        temporalVar["rcBufferOffsets"] = mpRcBufferOffsets;
        temporalVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        temporalVar["foundNeighborPixels"] = mpFoundNeighborPixels;

        temporalVar["prevCameraU"] = mPrevCameraU;
        temporalVar["prevCameraV"] = mPrevCameraV;
        temporalVar["prevCameraW"] = mPrevCameraW;
        temporalVar["prevJitterX"] = mPrevJitterX;
        temporalVar["prevJitterY"] = mPrevJitterY;

        ShaderVar prefixVar = bindSuffixResamplingVars(pRenderContext, mpSuffixResampling, "gSuffixResampling", pVBuffer, pMotionVectors, true, true);
        prefixVar["outColor"] = pOutputColor;
        prefixVar["rcBufferOffsets"] = mpRcBufferOffsets;
        prefixVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        prefixVar["foundNeighborPixels"] = mpFoundNeighborPixels;

        ShaderVar workloadVar;
        if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact)
        {
            workloadVar = bindSuffixResamplingVars(
                pRenderContext, mpSuffixProduceRetraceWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false, false
            );
            workloadVar["queue"]["counter"] = mpCounter;
            workloadVar["queue"]["workload"] = mpWorkload;
            workloadVar["foundNeighborPixels"] = mpFoundNeighborPixels;
        }

        ShaderVar retraceVar = bindSuffixResamplingVars(
            pRenderContext, mpSuffixRetrace, "gSuffixPathRetrace", pVBuffer, pMotionVectors, true, false
        );
        retraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVar["rcBufferOffsets"] = mpRcBufferOffsets;
        retraceVar["queue"]["counter"] = mpCounter;
        retraceVar["queue"]["workload"] = mpWorkload;
        retraceVar["foundNeighborPixels"] = mpFoundNeighborPixels;

        ShaderVar workloadVarTalbot;
        if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather)
        {
            workloadVarTalbot = bindSuffixResamplingVars(
                pRenderContext, mpSuffixProduceRetraceTalbotWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false, false
            );
            workloadVarTalbot["queue"]["counter"] = mpCounter;
            workloadVarTalbot["queue"]["workload"] = mpWorkload;
            workloadVarTalbot["queue"]["workloadExtra"] = mpWorkloadExtra;
            workloadVarTalbot["foundNeighborPixels"] = mpFoundNeighborPixels;
        }

        ShaderVar retraceVarTalbot = bindSuffixResamplingVars(
            pRenderContext, mpSuffixRetraceTalbot, "gSuffixPathRetrace", pVBuffer, pMotionVectors, true, false
        );
        retraceVarTalbot["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVarTalbot["rcBufferOffsets"] = mpRcBufferOffsets;
        retraceVarTalbot["queue"]["counter"] = mpCounter;
        retraceVarTalbot["queue"]["workload"] = mpWorkload;
        retraceVarTalbot["queue"]["workloadExtra"] = mpWorkloadExtra;
        retraceVarTalbot["foundNeighborPixels"] = mpFoundNeighborPixels;

        ShaderVar prefixWorkloadVar;
        if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact)
        {
            prefixWorkloadVar = bindPrefixResamplingVars(
                pRenderContext, mpPrefixProduceRetraceWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false
            );
            prefixWorkloadVar["queue"]["counter"] = mpCounter;
            prefixWorkloadVar["queue"]["workload"] = mpWorkload;
        }

        ShaderVar prefixRetraceVar = bindPrefixResamplingVars(
            pRenderContext, mpPrefixRetrace, "gPrefixPathRetrace", pVBuffer, pMotionVectors, true
        );
        prefixRetraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        prefixRetraceVar["rcBufferOffsets"] = mpRcBufferOffsets;
        prefixRetraceVar["queue"]["counter"] = mpCounter;
        prefixRetraceVar["queue"]["workload"] = mpWorkload;
        prefixRetraceVar["prefixReservoirs"] = mpPrefixReservoirs;
        prefixRetraceVar["prevPrefixReservoirs"] = mpPrevPrefixReservoirs;
        prefixRetraceVar["prefixTotalLengthBuffer"] = mpPrefixL2LengthBuffer; // abuse the storage for this

        // try to re-bind the correct value for a term used to offset RNG
        int numRounds = mOptions.subpathSetting.suffixSpatialReuseRounds;
        int numRoundsForComputeRNG = numRounds + 1 + (hasTemporalReuse ? 1 : 0);
        spatialVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        temporalVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        prefixVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;

        if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact)
        {
            workloadVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
            prefixWorkloadVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
            if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather)
                workloadVarTalbot["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        }
        retraceVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        prefixRetraceVar["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather)
            retraceVarTalbot["restir"]["suffixSpatialRounds"] = numRoundsForComputeRNG;

        if (mOptions.subpathSetting.adaptivePrefixLength)
        {
            if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact)
            {
                FALCOR_PROFILE("ProducePrefixWorkload");

                if (mpCounter)
                    pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                prefixWorkloadVar["prevReservoirs"] = mpPrevSuffixReservoirs;
                const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                mpPrefixProduceRetraceWorkload->execute(
                    pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
                );
            }

            {
                FALCOR_PROFILE("PrefixRetrace");
                prefixRetraceVar["prevReservoirs"] = mpPrevSuffixReservoirs;

                mpPrefixRetrace->execute(pRenderContext,
                    mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Naive ? mFrameDim.x : 2 * mFrameDim.x * mFrameDim.y,
                    mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Naive ? mFrameDim.y : 1, 1);
            }
        }

        {
            FALCOR_PROFILE("PrefixResampling");

            presamplingVar["reservoirs"] = mpReservoirs;
            presamplingVar["prevReservoirs"] = mpPrevSuffixReservoirs;
            presamplingVar["prefixSearchKeys"] = mpFinalGatherSearchKeys;
            presamplingVar["searchPointBoundingBoxBuffer"] = mpSearchPointBoundingBoxBuffer;
            presamplingVar["prefixTotalLengthBuffer"] = mpPrefixL2LengthBuffer;
            presamplingVar["screenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(mFrameDim.y);

            mpPrefixResampling->execute(
                pRenderContext, mFrameDim.x, mFrameDim.y, 1
            );
        }

        if (!mpSearchASBuilder)
        {
            mpSearchASBuilder = BoundingBoxAccelerationStructureBuilder::Create(mpSearchPointBoundingBoxBuffer);
        }

        if (!mResetTemporalReservoirs)
        {
            FALCOR_PROFILE("BuildSearchAS");
            uint numSearchPoints = mFrameDim.x * mFrameDim.y;
            mpSearchASBuilder->BuildAS(pRenderContext, numSearchPoints, 1);
        }

        // trace an additional path
        {
            FALCOR_PROFILE("TraceNewSuffixes");

            // Bind global resources.
            auto var = mpTraceNewSuffixes->getRootVar();
            mpScene->setRaytracingShaderData(pRenderContext, var);
            mpPixelDebug->prepareProgram(mpTraceNewSuffixes->getProgram(), var);
            mpPixelStats->prepareProgram(mpTraceNewSuffixes->getProgram(), mpTraceNewSuffixes->getRootVar());

            // Bind the path tracer.
            var["gPathTracer"] = mpPathTracerBlock;
            var["gScheduler"]["prefixGbuffer"] = mpPrefixGBuffer;
            var["gScheduler"]["pathReservoirs"] = mpReservoirs;
            // Full screen dispatch.
            mpTraceNewSuffixes->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }

        int numLevels = 1;

        for (int iter = 0; iter < numLevels; iter++)
        {
            int numRounds = mOptions.subpathSetting.suffixSpatialReuseRounds;
            // the actual rounds used
            numRounds = mOptions.subpathSetting.suffixTemporalReuse ? numRounds + 1 : numRounds;

            for (int i = 0; i < numRounds; i++)
            {
                bool isCurrentPassTemporal = mOptions.subpathSetting.suffixTemporalReuse && i == 0;
                Buffer::SharedPtr& pPrevSuffixReservoirs = (isCurrentPassTemporal || !mpScene->freeze) ? mpPrevSuffixReservoirs : mpTempReservoirs;

                if (!isCurrentPassTemporal)
                {
                    std::swap(mpReservoirs, pPrevSuffixReservoirs);
                }

                if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact)
                {
                    FALCOR_PROFILE(isCurrentPassTemporal ? "TemporalSuffixProduceRetraceWorkload" : "SpatialSuffixProduceRetraceWorkload");

                    if (mpCounter)
                        pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                    workloadVar["reservoirs"] = mpReservoirs;
                    workloadVar["prevReservoirs"] = pPrevSuffixReservoirs;
                    workloadVar["suffixReuseRoundId"] = i;
                    workloadVar["curPrefixLength"] = numLevels - iter;

                    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                    mpSuffixProduceRetraceWorkload->execute(
                        pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
                    );
                }

                {
                    FALCOR_PROFILE(isCurrentPassTemporal ? "TemporalSuffixRetrace" : "SpatialSuffixRetrace");

                    retraceVar["reservoirs"] = mpReservoirs;
                    retraceVar["prevReservoirs"] = pPrevSuffixReservoirs;
                    retraceVar["suffixReuseRoundId"] = i;

                    if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Naive)
                        mpSuffixRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                    else
                        mpSuffixRetrace->execute(pRenderContext, 2 * (isCurrentPassTemporal ? 1 : mOptions.subpathSetting.suffixSpatialNeighborCount) * mFrameDim.x * mFrameDim.y, 1, 1);
                }


                {
                    FALCOR_PROFILE(isCurrentPassTemporal ? "TemporalSuffixResampling" : "SpatialSuffixResampling");

                    ShaderVar& tempVar = isCurrentPassTemporal ? temporalVar : spatialVar;
                    ComputePass::SharedPtr& tempPass = isCurrentPassTemporal ? mpSuffixTemporalResampling : mpSuffixSpatialResampling;

                    tempVar["reservoirs"] = mpReservoirs;
                    tempVar["prevReservoirs"] = pPrevSuffixReservoirs;
                    tempVar["suffixReuseRoundId"] = i;
                    tempVar["curPrefixLength"] = numLevels - iter;
                    tempVar["vbuffer"] = pVBuffer;

                    tempPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                }
            }

            mOptions.subpathSetting.suffixTemporalReuse = hasTemporalReuse;


            //  generate multiple suffixes
            Buffer::SharedPtr& pPrevSuffixReservoirs = !mpScene->freeze ? mpPrevSuffixReservoirs : mpTempReservoirs;
            std::swap(mpReservoirs, pPrevSuffixReservoirs);

            for (int integrationPrefixId = 0; integrationPrefixId < mOptions.subpathSetting.numIntegrationPrefixes; integrationPrefixId++)
            {
                bool hasCanonicalSuffix =
                    mOptions.subpathSetting.generateCanonicalSuffixForEachPrefix ? true : integrationPrefixId == 0;

                // we borrow the prefix of integrationPrefixId 0 from before
                {
                    // trace new prefixes
                    FALCOR_PROFILE("TraceNewPrefixes");

                    // Bind global resources.
                    auto var = mpTraceNewPrefixes->getRootVar();
                    mpScene->setRaytracingShaderData(pRenderContext, var);
                    mpPixelDebug->prepareProgram(mpTraceNewPrefixes->getProgram(), var);
                    mpPixelStats->prepareProgram(mpTraceNewPrefixes->getProgram(), var);
                    // Bind the path tracer.
                    var["gPathTracer"] = mpPathTracerBlock;
                    var["gScheduler"]["integrationPrefixId"] = integrationPrefixId;
                    var["gScheduler"]["shouldGenerateSuffix"] = hasCanonicalSuffix;
                    // Full screen dispatch.
                    mpTraceNewPrefixes->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                }

                // stream prefixes
                {
                    FALCOR_PROFILE("FinalGather");

                    {
                        FALCOR_PROFILE("FinalGatherNeighborSearch");

                        mpPrefixNeighborSearch["gScene"] = mpScene->getParameterBlock();
                        auto rootVar = mpPrefixNeighborSearch->getRootVar();
                        mpPixelDebug->prepareProgram(mpPrefixNeighborSearch->getProgram(), rootVar);
                        auto var = rootVar["CB"]["gPrefixNeighborSearch"];

                        var["neighborOffsets"] = mpNeighborOffsets;
                        var["motionVectors"] = pMotionVectors;
                        var["params"].setBlob(mPathTracerParams);
                        setShaderData(var["restir"]);
                        var["prefixGBuffer"] = mpScratchPrefixGBuffer;
                        var["prevPrefixGBuffer"] = mpPrefixGBuffer;
                        var["foundNeighborPixels"] = mpFoundNeighborPixels;
                        var["integrationPrefixId"] = integrationPrefixId;
                        var["prefixSearchKeys"] = mpFinalGatherSearchKeys;
                        var["hasSearchPointAS"] = !mResetTemporalReservoirs;
                        var["searchPointBoundingBoxBuffer"] = mpSearchPointBoundingBoxBuffer;

                        if (mpSearchASBuilder && !mResetTemporalReservoirs)
                            mpSearchASBuilder->SetRaytracingShaderData(var, "gSearchPointAS", 1u);

                        mpPrefixNeighborSearch->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                    }

                    ComputePass::SharedPtr pFinalGatherRetraceProduceWorkload = mOptions.subpathSetting.useTalbotMISForGather ?
                        mpSuffixProduceRetraceTalbotWorkload : mpSuffixProduceRetraceWorkload;

                    if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Compact)
                    {
                        FALCOR_PROFILE("FinalGatherProduceRetraceWorkload");

                        if (mpCounter)
                            pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                        ShaderVar& var = mOptions.subpathSetting.useTalbotMISForGather ? workloadVarTalbot : workloadVar;

                        var["prevReservoirs"] = pPrevSuffixReservoirs;
                        var["suffixReuseRoundId"] = -1;
                        var["integrationPrefixId"] = integrationPrefixId;

                        const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                        pFinalGatherRetraceProduceWorkload->execute(
                            pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
                        );
                    }

                    ComputePass::SharedPtr pSuffixRetrace = mOptions.subpathSetting.useTalbotMISForGather ?
                        mpSuffixRetraceTalbot: mpSuffixRetrace;

                    {
                        FALCOR_PROFILE("FinalGatherSuffixRetrace");

                        ShaderVar& var = mOptions.subpathSetting.useTalbotMISForGather ? retraceVarTalbot : retraceVar;

                        var["prevReservoirs"] = pPrevSuffixReservoirs;
                        var["suffixReuseRoundId"] = -1;
                        var["integrationPrefixId"] = integrationPrefixId;

                        int multiplier = mOptions.subpathSetting.useTalbotMISForGather ? mOptions.subpathSetting.finalGatherSuffixCount + 1 : 2;

                        if (mOptions.retraceScheduleType == ConditionalReSTIR::RetraceScheduleType::Naive)
                            pSuffixRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                        else
                            pSuffixRetrace->execute(pRenderContext, multiplier * mOptions.subpathSetting.finalGatherSuffixCount * mFrameDim.x * mFrameDim.y, 1, 1);
                    }

                    {
                        FALCOR_PROFILE("FinalGatherIntegration");

                        prefixVar["reservoirs"] = mpReservoirs;
                        prefixVar["prevReservoirs"] = pPrevSuffixReservoirs;
                        prefixVar["suffixReuseRoundId"] = -1;
                        prefixVar["prefixReservoirs"] = mpPrefixReservoirs;
                        prefixVar["curPrefixLength"] = numLevels - iter;
                        prefixVar["integrationPrefixId"] = integrationPrefixId;
                        prefixVar["hasCanonicalSuffix"] = hasCanonicalSuffix;

                        mpSuffixResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                    }
                }
            }
        }

        mResetTemporalReservoirs = false;

        // prepare temporal data
        if (!mpScene->freeze)
        {
            if (mpTemporalVBuffer)
                pRenderContext->copyResource(mpTemporalVBuffer.get(), pVBuffer.get());
            mPrevCameraU = mpScene->getCamera()->getData().cameraU;
            mPrevCameraV = mpScene->getCamera()->getData().cameraV;
            mPrevCameraW = mpScene->getCamera()->getData().cameraW;
            mPrevJitterX = mpScene->getCamera()->getData().jitterX;
            mPrevJitterY = mpScene->getCamera()->getData().jitterY;
        }
    }

    ShaderVar ConditionalReSTIRPass::bindSuffixResamplingVars(RenderContext* pRenderContext,
        ComputePass::SharedPtr pPass, std::string cbName, const Texture::SharedPtr& pVBuffer, const Texture::SharedPtr& pMotionVectors, bool bindPathTracer, bool bindVBuffer)
    {
        pPass["gScene"] = mpScene->getParameterBlock();
        auto rootVar = pPass->getRootVar();

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["neighborOffsets"] = mpNeighborOffsets;

        var["motionVectors"] = pMotionVectors;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restir"]);

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = !mpScene->freeze ? mpPrevSuffixReservoirs : mpTempReservoirs; // doesn't matter, this will be binded differently for different passes

        var["prefixGBuffer"] = mpPrefixGBuffer;
        var["prevPrefixGBuffer"] = mpPrevPrefixGBuffer;

        var["neighborValidMask"] = mpNeighborValidMaskBuffer;

        if (bindVBuffer)
        {
            var["vbuffer"] = pVBuffer;
            var["temporalVbuffer"] = mpTemporalVBuffer;
        }
        return var;
    }


    ShaderVar ConditionalReSTIRPass::bindSuffixResamplingOneVars(RenderContext* pRenderContext,
        ComputePass::SharedPtr pPass, std::string cbName, const Texture::SharedPtr& pMotionVectors, bool bindPathTracer)
    {
        pPass["gScene"] = mpScene->getParameterBlock();
        auto rootVar = pPass->getRootVar();

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["motionVectors"] = pMotionVectors;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restir"]);

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = mpPrevSuffixReservoirs;

        var["prefixGBuffer"] = mpPrefixGBuffer;

        return var;
    }


    ShaderVar ConditionalReSTIRPass::bindPrefixResamplingVars(RenderContext* pRenderContext, ComputePass::SharedPtr pPass, std::string cbName, const Texture::SharedPtr& pVBuffer, const Texture::SharedPtr& pMotionVectors, bool bindPathTracer)
    {
        pPass["gScene"] = mpScene->getParameterBlock();
        auto rootVar = pPass->getRootVar();

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["vbuffer"] = pVBuffer;
        var["temporalVbuffer"] = mpTemporalVBuffer;

        var["motionVectors"] = pMotionVectors;

        var["prevCameraU"] = mPrevCameraU;
        var["prevCameraV"] = mPrevCameraV;
        var["prevCameraW"] = mPrevCameraW;
        var["prevJitterX"] = mPrevJitterX;
        var["prevJitterY"] = mPrevJitterY;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restir"]);

        var["neighborValidMask"] = mpNeighborValidMaskBuffer;

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        return var;
    }


    Texture::SharedPtr ConditionalReSTIRPass::createNeighborOffsetTexture(uint32_t sampleCount)
    {
        std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
        const int R = 254;
        const float phi2 = 1.f / 1.3247179572447f;
        float u = 0.5f;
        float v = 0.5f;
        for (uint32_t index = 0; index < sampleCount * 2;)
        {
            u += phi2;
            v += phi2 * phi2;
            if (u >= 1.f) u -= 1.f;
            if (v >= 1.f) v -= 1.f;

            float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
            if (rSq > 0.25f) continue;

            offsets[index++] = int8_t((u - 0.5f) * R);
            offsets[index++] = int8_t((v - 0.5f) * R);
        }

        return Texture::create1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
    }

    void ConditionalReSTIRPass::scriptBindings(pybind11::module& m)
    {
        // This library can be used from multiple render passes. If already registered, return immediately.
        if (pybind11::hasattr(m, "ConditionalReSTIROptions")) return;

        ScriptBindings::SerializableStruct<ConditionalReSTIR::SubpathReuseSettings> subpathsettings(m, "SubpathSettings");
#define field(f_) field(#f_, &ConditionalReSTIR::SubpathReuseSettings::f_)
        subpathsettings.field(useMMIS);

        subpathsettings.field(suffixSpatialNeighborCount);
        subpathsettings.field(suffixSpatialReuseRadius);
        subpathsettings.field(suffixSpatialReuseRounds);
        subpathsettings.field(suffixTemporalReuse);
        subpathsettings.field(temporalHistoryLength);
        // TODO: add fields
        subpathsettings.field(finalGatherSuffixCount);
        subpathsettings.field(prefixNeighborSearchRadius);
        subpathsettings.field(prefixNeighborSearchNeighborCount);

        subpathsettings.field(numIntegrationPrefixes);
        subpathsettings.field(generateCanonicalSuffixForEachPrefix);

#undef field


        ScriptBindings::SerializableStruct<Options> options(m, "ConditionalReSTIROptions");
#define field(f_) field(#f_, &Options::f_)

        options.field(subpathSetting);
        options.field(shiftMappingSettings);

#undef field

        ScriptBindings::SerializableStruct<ConditionalReSTIR::ShiftMappingSettings> shiftmappingsettings(m, "ShiftMappingSettings");
#define field(f_) field(#f_, &ConditionalReSTIR::ShiftMappingSettings::f_)
        shiftmappingsettings.field(localStrategyType);
        shiftmappingsettings.field(specularRoughnessThreshold);
        shiftmappingsettings.field(nearFieldDistanceThreshold);
#undef field
    }
}
