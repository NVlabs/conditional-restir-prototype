/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#pragma once
#include "Utils/Sampling/AliasTable.h"
#include "Utils/Debug/PixelDebug.h"
#include "Rendering/Utils/PixelStats.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/Scene.h"
#include "Scene/Lights/LightCollection.h"
#include "Scene/Lights/Light.h"
#include "ConditionalReSTIR.slang"
#include "Params.slang"
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>
#include "Scene/BoundingBoxAccelerationStructureBuilder.h"

namespace Falcor
{
    /** Implementation of Conditional ReSTIR (final gather version of ReSTIR)
        based on
        "Conditional Resampled Importance Sampling and ReSTIR"
        [Kettunen et al. 2023].
    */
    class FALCOR_API ConditionalReSTIRPass
    {
    public:
        using SharedPtr = std::shared_ptr<ConditionalReSTIRPass>;

        /** Configuration options.
        */
        struct Options
        {
            // Common Options for ReSTIR DI and GI.
 
            // Temporal resampling options.
            bool temporalUpdateForDynamicScene = false;

            // TODO: MIS option

            // Options for ReSTIR.
            ConditionalReSTIR::ShiftMappingSettings shiftMappingSettings;

            uint32_t reservoirCountPerPixel = 1;                ///< Number of reservoirs per pixel.

            // static params
            ConditionalReSTIR::ShiftMapping shiftMapping = ConditionalReSTIR::ShiftMapping::Hybrid;

            bool useReservoirCompression = true;

            uint32_t minimumPrefixLength = 1;

            ConditionalReSTIR::SubpathReuseSettings subpathSetting;

            // subpath reuse general settings

            bool visualizeFireflies = false;

            bool usePrevFrameSceneData = false;

            ConditionalReSTIR::RetraceScheduleType retraceScheduleType = ConditionalReSTIR::RetraceScheduleType::Compact;

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}
        };

        // static params shared with internal path tracer
        struct SharedStaticParams
        {
            uint32_t    samplesPerPixel;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
            uint32_t    maxSurfaceBounces;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth. This will be initialized at startup.
            bool        useNEE;                              ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        };

        /** Create a new instance of the ReSTIR sampler.
            \param[in] pScene Scene.
            \param[in] options Configuration options.
        */
        static SharedPtr create(const Scene::SharedPtr& pScene, const Program::DefineList& ownerDefines, const Options& options, const PixelStats::SharedPtr& pPixelStats);

        /** Get a list of shader defines for using the ReSTIR sampler.
            \return Returns a list of defines.
        */
        Program::DefineList getDefines() const;

        /** Bind the ReSTIR sampler to a given shader var.
            \param[in] var The shader variable to set the data into.
        */
        void setShaderData(const ShaderVar& var) const;

        void setReservoirData(const ShaderVar& var) const;

        /** Render the GUI.
            \return True if options were changed, false otherwise.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Returns the current configuration.
        */
        Options& getOptions()  { return mOptions; }

        /** Set the configuration.
        */
        void setOptions(const Options& options);

        /** Begin a frame.
            Must be called once at the beginning of each frame.
            \param[in] pRenderContext Render context.
            \param[in] frameDim Current frame dimension.
        */
        void beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile);

        /** End a frame.
            Must be called one at the end of each frame.
            \param[in] pRenderContext Render context.
        */
        void endFrame(RenderContext* pRenderContext);


        /** Get the debug output texture.
            \return Returns the debug output texture.
        */
        const Texture::SharedPtr& getDebugOutputTexture() const { return mpDebugOutputTexture; }

        /** Get the pixel debug component.
            \return Returns the pixel debug component.
        */
        const PixelDebug::SharedPtr& getPixelDebug() const { return mpPixelDebug; }

        /** Register script bindings.
        */
        static void scriptBindings(pybind11::module& m);

        void setPathTracerParams(int useFixedSeed, uint fixedSeed,
            float lodBias, float specularRoughnessThreshold, uint2 frameDim, uint2 screenTiles, uint frameCount, uint seed, int samplesPerPixel, int DIMode);

        void setOwnerDefines(Program::DefineList defines);

        void setSharedStaticParams(uint32_t samplesPerPixel, uint32_t maxSurfaceBounces, bool useNEE);

        void createPathTracerBlock();

        ParameterBlock::SharedPtr getPathTracerBlock();

        void updatePrograms();

        void suffixResamplingPass(
            RenderContext* pRenderContext,
            const Texture::SharedPtr& pVBuffer,
            const Texture::SharedPtr& pMotionVectors,
            const Texture::SharedPtr& pOutputColor
        );

        bool needResetTemporalHistory() { return mResetTemporalReservoirs;  }

    private:
        ConditionalReSTIRPass(const Scene::SharedPtr& pScene, const Program::DefineList& ownerDefines, const Options& options, const PixelStats::SharedPtr& pPixelStats);

        void prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles);

        void createOrDestroyBuffer(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition=true);
        void createOrDestroyBufferWithCounter(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition = true);
        void createOrDestroyBufferNoReallocate(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition = true);
        void createOrDestroyBufferWithCounterNoReallocate(Buffer::SharedPtr& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition = true);

        void createOrDestroyRawBuffer(Buffer::SharedPtr& pBuffer, size_t requiredSize, bool keepCondition=true);

        Falcor::ShaderVar bindSuffixResamplingVars(RenderContext* pRenderContext, ComputePass::SharedPtr pPass, std::string cbName, const Texture::SharedPtr& pVBuffer, const Texture::SharedPtr& pMotionVectors, bool bindPathTracer, bool bindVBuffer);
        Falcor::ShaderVar bindPrefixResamplingVars(RenderContext* pRenderContext, ComputePass::SharedPtr pPass, std::string cbName, const Texture::SharedPtr& pVBuffer, const Texture::SharedPtr& pMotionVectors, bool bindPathTracer);
        Falcor::ShaderVar bindSuffixResamplingOneVars(RenderContext* pRenderContext, ComputePass::SharedPtr pPass, std::string cbName, const Texture::SharedPtr& pMotionVectors, bool bindPathTracer);
        /** Create a 1D texture with random offsets within a unit circle around (0,0).
            The texture is RG8Snorm for compactness and has no mip maps.
            \param[in] sampleCount Number of samples in the offset texture.
        */
        Texture::SharedPtr createNeighborOffsetTexture(uint32_t sampleCount);

        void createComputePass(ComputePass::SharedPtr& pPass, std::string shaderFile, Program::DefineList defines, Program::Desc desc, std::string entryFunction="");

        Scene::SharedPtr mpScene;                           ///< Scene.
        Options mOptions;                                   ///< Configuration options.
        SharedStaticParams mStaticParams;

        ReSTIRPathTracerParams                mPathTracerParams;                    ///< agrees with that in InlinePathTracer

        Program::DefineList mOwnerDefines;                   ///< Share defines with inline path tracer

        std::mt19937 mRng;                                  ///< Random generator.


        PixelStats::SharedPtr           mpPixelStats;               ///< Utility class for collecting pixel stats. (shared with host renderpass)

        PixelDebug::SharedPtr mpPixelDebug;                 ///< Pixel debug component.

        uint2 mFrameDim = uint2(0);                         ///< Current frame dimensions.
        uint32_t mFrameIndex = 0;                           ///< Current frame index.

        ComputePass::SharedPtr mpReflectTypes;              ///< Pass for reflecting types.

        // ReSTIR passes.

        ComputePass::SharedPtr mpPrefixProduceRetraceWorkload;
        ComputePass::SharedPtr mpPrefixRetrace;
        ComputePass::SharedPtr mpSuffixSpatialResampling;
        ComputePass::SharedPtr mpSuffixTemporalResampling;
        ComputePass::SharedPtr mpSuffixResampling;
        ComputePass::SharedPtr mpSuffixRetrace;
        ComputePass::SharedPtr mpSuffixProduceRetraceWorkload;
        ComputePass::SharedPtr mpSuffixRetraceTalbot;
        ComputePass::SharedPtr mpSuffixProduceRetraceTalbotWorkload;
        ComputePass::SharedPtr mpPrefixResampling;
        ComputePass::SharedPtr mpTraceNewSuffixes;
        ComputePass::SharedPtr mpTraceNewPrefixes;
        ComputePass::SharedPtr mpPrefixNeighborSearch;

        ParameterBlock::SharedPtr       mpPathTracerBlock;          ///< Parameter block for the path tracer.

        Buffer::SharedPtr mpPrefixGBuffer;          ///< Buffer containing the current sample's path vertices
        Buffer::SharedPtr mpPrevPrefixGBuffer;

        Buffer::SharedPtr mpPrefixReservoirs;
        Buffer::SharedPtr mpPrevPrefixReservoirs;

        Buffer::SharedPtr mpScratchPrefixGBuffer;

        Buffer::SharedPtr mpScratchReservoirs;              ///< Buffer containing the temporary reservoirs. // can also hold firefly reservoirs/prev suffix reservoirs
        Buffer::SharedPtr mpReservoirs;                     ///< Buffer containing the current reservoirs.
        Buffer::SharedPtr mpPrefixPathReservoirs;
        Buffer::SharedPtr mpPrefixThroughputs;
        Buffer::SharedPtr mpPrevReservoirs;                 ///< Buffer containing the previous reservoirs.
        Buffer::SharedPtr mpPrevSuffixReservoirs;           ///< Buffer containing previous suffix reservoirs.
        Buffer::SharedPtr mpFoundNeighborPixels;   

        Buffer::SharedPtr mpTempReservoirs;                 ///< can hold both initial sampling results and firefly path reserovirs
        Buffer::SharedPtr mpReconnectionDataBuffer;          ///< Buffer containing the reconnection data for retrace result.
        Buffer::SharedPtr mpRcBufferOffsets;
        Buffer::SharedPtr mpNeighborValidMaskBuffer;

        Buffer::SharedPtr mpFinalGatherSearchKeys;

        Buffer::SharedPtr               mpWorkload;             ///< Paths starting from primary hits on general materials (all types).
        Buffer::SharedPtr               mpWorkloadExtra;             ///< Paths starting from primary hits on general materials (all types).
        Buffer::SharedPtr               mpCounter;                 ///< Atomic counters (32-bit).

        Texture::SharedPtr mpDebugOutputTexture;            ///< Debug output texture.
        Texture::SharedPtr mpNeighborOffsets;               ///< 1D texture containing neighbor offsets within a unit circle.

        Buffer::SharedPtr mpSearchPointBoundingBoxBuffer;
        Buffer::SharedPtr mpPrefixL2LengthBuffer;
        BoundingBoxAccelerationStructureBuilder::SharedPtr mpSearchASBuilder;

        // temporal data.
        float3 mPrevCameraU;
        float3 mPrevCameraV;
        float3 mPrevCameraW;
        float mPrevJitterX;
        float mPrevJitterY;
        Texture::SharedPtr mpTemporalVBuffer;

public:
        bool mRecompile = true;                             ///< Recompile programs on next frame if set to true.
        bool mReallocate = true;                            ///< Reallocate the reservoirs since sizes change
        bool mResetTemporalReservoirs = true;               ///< Reset temporal reservoir buffer on next frame if set to true.

    };
}
