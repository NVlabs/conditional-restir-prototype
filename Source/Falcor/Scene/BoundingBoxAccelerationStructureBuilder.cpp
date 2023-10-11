/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "BoundingBoxAccelerationStructureBuilder.h"
#include "Core/API/GFX/GFXAPI.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    BoundingBoxAccelerationStructureBuilder::SharedPtr BoundingBoxAccelerationStructureBuilder::Create(Buffer::SharedPtr pBoundingBoxBuffer)
    {
        BoundingBoxAccelerationStructureBuilder::SharedPtr result = BoundingBoxAccelerationStructureBuilder::SharedPtr(new BoundingBoxAccelerationStructureBuilder());
        result->m_BoundingBoxBuffer = pBoundingBoxBuffer;
        return result;
    }

    void BoundingBoxAccelerationStructureBuilder::BuildAS(RenderContext* pContext, uint32_t boxCount, uint32_t rayTypeCount)
    {
        if (mRebuildBlas)
        {
            InitGeomDesc(boxCount);
            BuildBlas(pContext);
            BuildTlas(pContext, rayTypeCount, true);
        }
        else //just update
        {
            BuildBlas(pContext);
        }
    }

    void BoundingBoxAccelerationStructureBuilder::SetRaytracingShaderData(const ShaderVar& var, const std::string name, uint32_t rayTypeCount)
    {
        auto tlasIt = mTlasCache.find(rayTypeCount);
        FALCOR_ASSERT(tlasIt != mTlasCache.end() && tlasIt->second.pTlasObject)
        var[name].setAccelerationStructure(tlasIt->second.pTlasObject);
    }

    void BoundingBoxAccelerationStructureBuilder::InitGeomDesc(uint32_t boxCount)
    {
        mBlasData.resize(1);
        mBlasObjects.resize(1);
        auto& blas = mBlasData[0];
        auto& geomDescs = blas.geomDescs;
        geomDescs.resize(1);

        {
            RtGeometryDesc& desc = geomDescs[0];
            desc.type = RtGeometryType::ProcedurePrimitives;
            desc.flags = RtGeometryFlags::Opaque;

            RtAABBDesc AABBDesc = {};
            AABBDesc.count = boxCount;
            D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE addressAndStride = {};
            AABBDesc.stride = 32;
            AABBDesc.data = m_BoundingBoxBuffer->getGpuAddress();

            desc.content.proceduralAABBs = AABBDesc;
        }
    }


    void BoundingBoxAccelerationStructureBuilder::invalidateTlasCache()
    {
        for (auto& tlas : mTlasCache)
        {
            tlas.second.pTlasObject = nullptr;
        }
    }

    void BoundingBoxAccelerationStructureBuilder::BuildBlas(RenderContext* pContext)
    {
        pContext->resourceBarrier(m_BoundingBoxBuffer.get(), Resource::State::NonPixelShader);

        bool useCompaction = true;
        bool hasProceduralPrimitives = true;
        bool useRefit = false;

        if (!useRefit) useCompaction = false;

        if (mRebuildBlas)
        {
            // Invalidate any previous TLASes as they won't be valid anymore.
            invalidateTlasCache();

            if (mBlasData.empty())
            {
                logInfo("Skipping BLAS build due to no geometries");
                mBlasObjects.clear();
            }
            else
            {
                logInfo("Initiating BLAS build for {} mesh groups", mBlasData.size());

                // Compute pre-build info per BLAS and organize the BLASes into groups
                // in order to limit GPU memory usage during BLAS build.

                // Compute the required maximum size of the result and scratch buffers.
                uint64_t resultByteSize = 0;
                uint64_t scratchByteSize = 0;
                size_t maxBlasCount = 1;

                {
                    auto& blas = mBlasData[0];

                    // Setup build parameters.
                    RtAccelerationStructureBuildInputs& inputs = blas.buildInputs;
                    inputs.kind = RtAccelerationStructureKind::BottomLevel;
                    inputs.descCount = (uint32_t)blas.geomDescs.size();
                    inputs.geometryDescs = blas.geomDescs.data();
                    inputs.flags = RtAccelerationStructureBuildFlags::None;
                    if (useCompaction)
                        inputs.flags |= RtAccelerationStructureBuildFlags::AllowCompaction;
                    if (useRefit)
                        inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;
                    inputs.flags |= RtAccelerationStructureBuildFlags::PreferFastBuild;

                    // Get prebuild info.
                    blas.prebuildInfo = RtAccelerationStructure::getPrebuildInfo(inputs);
                    // Figure out the padded allocation sizes to have proper alignment.
                    FALCOR_ASSERT(blas.prebuildInfo.resultDataMaxSize > 0);
                    resultByteSize = align_to(kAccelerationStructureByteAlignment, blas.prebuildInfo.resultDataMaxSize);

                    scratchByteSize = std::max(blas.prebuildInfo.scratchDataSize, blas.prebuildInfo.updateScratchDataSize);
                    scratchByteSize = align_to(kAccelerationStructureByteAlignment, scratchByteSize);
                }

                FALCOR_ASSERT(resultByteSize > 0 && scratchByteSize > 0);

                logInfo("BLAS build result buffer size: {}", formatByteSize(resultByteSize));
                logInfo("BLAS build scratch buffer size: {}", formatByteSize(scratchByteSize));

                // Allocate result and scratch buffers.
                // The scratch buffer we'll retain because it's needed for subsequent rebuilds and updates.
                // TODO: Save memory by reducing the scratch buffer to the minimum required for the dynamic objects.
                if (mpBlasScratch == nullptr || mpBlasScratch->getSize() < scratchByteSize)
                {
                    mpBlasScratch = Buffer::create(scratchByteSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                    mpBlasScratch->setName("Scene::mpBlasScratch");
                }

                Buffer::SharedPtr pResultBuffer = Buffer::create(resultByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                FALCOR_ASSERT(pResultBuffer && mpBlasScratch);

                // Create post-build info pool for readback.
                RtAccelerationStructurePostBuildInfoPool::Desc compactedSizeInfoPoolDesc;
                compactedSizeInfoPoolDesc.queryType = RtAccelerationStructurePostBuildInfoQueryType::CompactedSize;
                compactedSizeInfoPoolDesc.elementCount = (uint32_t)maxBlasCount;
                RtAccelerationStructurePostBuildInfoPool::SharedPtr compactedSizeInfoPool = RtAccelerationStructurePostBuildInfoPool::create(compactedSizeInfoPoolDesc);

                RtAccelerationStructurePostBuildInfoPool::Desc currentSizeInfoPoolDesc;
                currentSizeInfoPoolDesc.queryType = RtAccelerationStructurePostBuildInfoQueryType::CurrentSize;
                currentSizeInfoPoolDesc.elementCount = (uint32_t)maxBlasCount;
                RtAccelerationStructurePostBuildInfoPool::SharedPtr currentSizeInfoPool = RtAccelerationStructurePostBuildInfoPool::create(currentSizeInfoPoolDesc);

                {
                    // Allocate array to hold intermediate blases for the group.
                    std::vector<RtAccelerationStructure::SharedPtr> intermediateBlases(1);

                    // Insert barriers. The buffers are now ready to be written.
                    pContext->uavBarrier(pResultBuffer.get());
                    pContext->uavBarrier(mpBlasScratch.get());

                    // Reset the post-build info pools to receive new info.
                    compactedSizeInfoPool->reset(pContext);
                    currentSizeInfoPool->reset(pContext);

                    // Build the BLASes into the intermediate result buffer.
                    // We output post-build info in order to find out the final size requirements.
                    {
                        const auto& blas = mBlasData[0];

                        RtAccelerationStructure::Desc createDesc = {};
                        createDesc.setBuffer(pResultBuffer, 0, resultByteSize);
                        createDesc.setKind(RtAccelerationStructureKind::BottomLevel);
                        auto blasObject = RtAccelerationStructure::create(createDesc);
                        intermediateBlases[0] = blasObject;

                        RtAccelerationStructure::BuildDesc asDesc = {};
                        asDesc.inputs = blas.buildInputs;
                        asDesc.scratchData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
                        asDesc.dest = blasObject.get();

                        // Need to find out the post-build compacted BLAS size to know the final allocation size.
                        RtAccelerationStructurePostBuildInfoDesc postbuildInfoDesc = {};
                        if (useCompaction)
                        {
                            postbuildInfoDesc.type = RtAccelerationStructurePostBuildInfoQueryType::CompactedSize;
                            postbuildInfoDesc.index = (uint32_t)0;
                            postbuildInfoDesc.pool = compactedSizeInfoPool.get();
                        }
                        else
                        {
                            postbuildInfoDesc.type = RtAccelerationStructurePostBuildInfoQueryType::CurrentSize;
                            postbuildInfoDesc.index = (uint32_t)0;
                            postbuildInfoDesc.pool = currentSizeInfoPool.get();
                        }

                        pContext->buildAccelerationStructure(asDesc, 1, &postbuildInfoDesc);
                    }

                    // Read back the calculated final size requirements for each BLAS.

                    uint64_t finalByteSize = 0;

                    {
                        auto& blas = mBlasData[0];

                        // Check the size. Upon failure a zero size may be reported.
                        uint64_t byteSize = 0;
                        if (useCompaction)
                        {
                            byteSize = compactedSizeInfoPool->getElement(pContext, (uint32_t)0);
                        }
                        else
                        {
                            byteSize = currentSizeInfoPool->getElement(pContext, (uint32_t)0);
                            // For platforms that does not support current size query, use prebuild size.
                            if (byteSize == 0)
                            {
                                byteSize = blas.prebuildInfo.resultDataMaxSize;
                            }
                        }
                        FALCOR_ASSERT(byteSize <= blas.prebuildInfo.resultDataMaxSize);
                        if (byteSize == 0) throw RuntimeError("Acceleration structure build failed for BLAS index {}", 0);

                        blas.blasByteSize = align_to(kAccelerationStructureByteAlignment, byteSize);
                        blas.blasByteOffset = finalByteSize;
                        finalByteSize += blas.blasByteSize;
                    }
                    FALCOR_ASSERT(group.finalByteSize > 0);

                    logInfo("BLAS group " + std::to_string(0) + " final size: " + formatByteSize(finalByteSize));

                    // Allocate final BLAS buffer.
                    auto& pBlas = mpBlas;
                    if (pBlas == nullptr || pBlas->getSize() < finalByteSize)
                    {
                        pBlas = Buffer::create(finalByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                        pBlas->setName("BBBuilder::mBlasGroups[" + std::to_string(0) + "].pBlas");
                    }
                    else
                    {
                        // If we didn't need to reallocate, just insert a barrier so it's safe to use.
                        pContext->uavBarrier(pBlas.get());
                    }

                    // Insert barrier. The result buffer is now ready to be consumed.
                    // TOOD: This is probably not necessary since we flushed above, but it's not going to hurt.
                    pContext->uavBarrier(pResultBuffer.get());

                    // Compact/clone all BLASes to their final location.
                    {
                        auto& blas = mBlasData[0];

                        RtAccelerationStructure::Desc blasDesc = {};
                        blasDesc.setBuffer(pBlas, blas.blasByteOffset, blas.blasByteSize);
                        blasDesc.setKind(RtAccelerationStructureKind::BottomLevel);
                        mBlasObjects[0] = RtAccelerationStructure::create(blasDesc);

                        pContext->copyAccelerationStructure(
                            mBlasObjects[0].get(),
                            intermediateBlases[0].get(),
                            useCompaction ? RenderContext::RtAccelerationStructureCopyMode::Compact : RenderContext::RtAccelerationStructureCopyMode::Clone);
                    }

                    // Insert barrier. The BLAS buffer is now ready for use.
                    pContext->uavBarrier(pBlas.get());
                }
            }

            mRebuildBlas = false;
            return;
        }

        // If we get here, all BLASes have previously been built and compacted. We will:
        // - Skip the ones that have no animated geometries.
        // - Update or rebuild in-place the ones that are animated.
        FALCOR_ASSERT(!mRebuildBlas);

        // just refit or update
        {
            // Determine if any BLAS in the group needs to be updated.
            const auto& blas = mBlasData[0];

            // At least one BLAS in the group needs to be updated.
            // Insert barriers. The buffers are now ready to be written.
            auto& pBlas = mpBlas;
            FALCOR_ASSERT(pBlas && mpBlasScratch);
            pContext->uavBarrier(pBlas.get());
            pContext->uavBarrier(mpBlasScratch.get());

            // Iterate over all BLASes in group.

            // Rebuild/update BLAS.
            RtAccelerationStructure::BuildDesc asDesc = {};
            asDesc.inputs = blas.buildInputs;
            asDesc.scratchData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
            asDesc.dest = mBlasObjects[0].get();

            if (useRefit)
            {
                // Set source address to destination address to update in place.
                asDesc.source = asDesc.dest;
                asDesc.inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
            }
            else
            {
                // We'll rebuild in place. The BLAS should not be compacted, check that size matches prebuild info.
                FALCOR_ASSERT(blas.blasByteSize == blas.prebuildInfo.resultDataMaxSize);
            }
            pContext->buildAccelerationStructure(asDesc, 0, nullptr);

            // Insert barrier. The BLAS buffer is now ready for use.
            pContext->uavBarrier(pBlas.get());
        }
    }

    void BoundingBoxAccelerationStructureBuilder::FillInstanceDesc(std::vector<RtInstanceDesc>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry)
    {
        assert(mpBlas);
        instanceDescs.clear();
        uint32_t instanceContributionToHitGroupIndex = 0;
        uint32_t instanceId = 0;

        for (size_t i = 0; i < mBlasData.size(); i++)
        {
            RtInstanceDesc desc = {};
            desc.accelerationStructure = mpBlas->getGpuAddress() + mBlasData[i].blasByteOffset;
            desc.instanceMask = 0xFF;
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;
            instanceContributionToHitGroupIndex += rayCount;
            //desc.InstanceID = 0;
            glm::mat4 transform4x4 = glm::mat4(1.0f);
            std::memcpy(desc.transform, &transform4x4, sizeof(desc.transform));
            instanceDescs.push_back(desc);
        }
    }

    void BoundingBoxAccelerationStructureBuilder::BuildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry)
    {
        FALCOR_PROFILE("buildTlas");

        TlasData tlas;
        auto it = mTlasCache.find(1);
        if (it != mTlasCache.end()) tlas = it->second;

        // Prepare instance descs.
        // Note if there are no instances, we'll build an empty TLAS.
        FillInstanceDesc(mInstanceDescs, 1, perMeshHitEntry);

        RtAccelerationStructureBuildInputs inputs = {};
        inputs.kind = RtAccelerationStructureKind::TopLevel;
        inputs.descCount = (uint32_t)mInstanceDescs.size();
        inputs.flags = RtAccelerationStructureBuildFlags::None;

        // Add build flags for dynamic scenes if TLAS should be updating instead of rebuilt

        // On first build for the scene, create scratch buffer and cache prebuild info. As long as INSTANCE_DESC count doesn't change, we can reuse these
        if (mpTlasScratch == nullptr)
        {
            // Prebuild
            mTlasPrebuildInfo = RtAccelerationStructure::getPrebuildInfo(inputs);
            mpTlasScratch = Buffer::create(mTlasPrebuildInfo.scratchDataSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            mpTlasScratch->setName("Scene::mpTlasScratch");

            // #SCENE This isn't guaranteed according to the spec, and the scratch buffer being stored should be sized differently depending on update mode
            FALCOR_ASSERT(mTlasPrebuildInfo.updateScratchDataSize <= mTlasPrebuildInfo.scratchDataSize);
        }

        // Setup GPU buffers
        RtAccelerationStructure::BuildDesc asDesc = {};
        asDesc.inputs = inputs;

        // If first time building this TLAS
        if (tlas.pTlasObject == nullptr)
        {
            {
                // Allocate a new buffer for the TLAS only if the existing buffer isn't big enough.
                if (!tlas.pTlasBuffer || tlas.pTlasBuffer->getSize() < mTlasPrebuildInfo.resultDataMaxSize)
                {
                    tlas.pTlasBuffer = Buffer::create(mTlasPrebuildInfo.resultDataMaxSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                    tlas.pTlasBuffer->setName("Scene TLAS buffer");
                }
            }
            if (!mInstanceDescs.empty())
            {
                // Allocate a new buffer for the TLAS instance desc input only if the existing buffer isn't big enough.
                if (!tlas.pInstanceDescs || tlas.pInstanceDescs->getSize() < mInstanceDescs.size() * sizeof(RtInstanceDesc))
                {
                    tlas.pInstanceDescs = Buffer::create((uint32_t)mInstanceDescs.size() * sizeof(RtInstanceDesc), Buffer::BindFlags::None, Buffer::CpuAccess::Write, mInstanceDescs.data());
                    tlas.pInstanceDescs->setName("Scene instance descs buffer");
                }
                else
                {
                    tlas.pInstanceDescs->setBlob(mInstanceDescs.data(), 0, mInstanceDescs.size() * sizeof(RtInstanceDesc));
                }
            }

            RtAccelerationStructure::Desc asCreateDesc = {};
            asCreateDesc.setKind(RtAccelerationStructureKind::TopLevel);
            asCreateDesc.setBuffer(tlas.pTlasBuffer, 0, mTlasPrebuildInfo.resultDataMaxSize);
            tlas.pTlasObject = RtAccelerationStructure::create(asCreateDesc);
        }
        // Else update instance descs and barrier TLAS buffers
        else
        {
            FALCOR_ASSERT(mpAnimationController->hasAnimations() || mpAnimationController->hasAnimatedVertexCaches());
            pContext->uavBarrier(tlas.pTlasBuffer.get());
            pContext->uavBarrier(mpTlasScratch.get());
            if (tlas.pInstanceDescs)
            {
                FALCOR_ASSERT(!mInstanceDescs.empty());
                tlas.pInstanceDescs->setBlob(mInstanceDescs.data(), 0, inputs.descCount * sizeof(RtInstanceDesc));
            }
            asDesc.source = tlas.pTlasObject.get(); // Perform the update in-place
        }

        FALCOR_ASSERT(tlas.pTlasBuffer && tlas.pTlasBuffer->getApiHandle() && mpTlasScratch->getApiHandle());
        FALCOR_ASSERT(inputs.descCount == 0 || (tlas.pInstanceDescs && tlas.pInstanceDescs->getApiHandle()));

        asDesc.inputs.instanceDescs = tlas.pInstanceDescs ? tlas.pInstanceDescs->getGpuAddress() : 0;
        asDesc.scratchData = mpTlasScratch->getGpuAddress();
        asDesc.dest = tlas.pTlasObject.get();

        // Set the source buffer to update in place if this is an update
        if ((inputs.flags & RtAccelerationStructureBuildFlags::PerformUpdate) != RtAccelerationStructureBuildFlags::None)
        {
            asDesc.source = asDesc.dest;
        }

        // Create TLAS
        if (tlas.pInstanceDescs)
        {
            pContext->resourceBarrier(tlas.pInstanceDescs.get(), Resource::State::NonPixelShader);
        }
        pContext->buildAccelerationStructure(asDesc, 0, nullptr);
        pContext->uavBarrier(tlas.pTlasBuffer.get());

        mTlasCache[1] = tlas;
    }
}
