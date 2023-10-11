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
#pragma once

#include "Falcor.h"

namespace Falcor
{
    class BoundingBoxAccelerationStructureBuilder
    {
    public:
        using SharedPtr = std::shared_ptr<BoundingBoxAccelerationStructureBuilder>;

        static SharedPtr Create(Buffer::SharedPtr pBoundingBoxBuffer);

        void BuildAS(RenderContext* pContext, uint32_t boxCount, uint32_t rayTypeCount);

        void SetRaytracingShaderData(const ShaderVar& var, const std::string name, uint32_t rayTypeCount);

    private:

        void InitGeomDesc(uint32_t boxCount);

        void invalidateTlasCache();

        void BuildBlas(RenderContext* pContext);

        void FillInstanceDesc(std::vector<RtInstanceDesc>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry);

        void BuildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry);

        Buffer::SharedPtr m_BoundingBoxBuffer;

        struct BlasData
        {
            RtAccelerationStructurePrebuildInfo prebuildInfo = {};
            RtAccelerationStructureBuildInputs buildInputs = {};
            std::vector<RtGeometryDesc> geomDescs;

            uint64_t blasByteSize = 0;                      ///< Size of the final BLAS.
            uint64_t blasByteOffset = 0;                    ///< Offset into the BLAS buffer to where it is stored.
            uint64_t scratchByteOffset = 0;                 ///< Offset into the scratch buffer to use for rebuilds.
        };

        struct TlasData
        {
            RtAccelerationStructure::SharedPtr pTlasObject;
			Buffer::SharedPtr pTlasBuffer;
			Buffer::SharedPtr pInstanceDescs;               ///< Buffer holding instance descs for the TLAS
        };

        std::vector<RtInstanceDesc> mInstanceDescs; ///< Shared between TLAS builds to avoid reallocating CPU memory
        std::unordered_map<uint32_t, TlasData> mTlasCache;  ///< Top Level Acceleration Structure for scene data cached per shader ray count
        Buffer::SharedPtr mpTlasScratch;                    ///< Scratch buffer used for TLAS builds. Can be shared as long as instance desc count is the same, which for now it is.
        RtAccelerationStructurePrebuildInfo mTlasPrebuildInfo; ///< This can be reused as long as the number of instance descs doesn't change.

        std::vector<BlasData> mBlasData;
        std::vector<RtAccelerationStructure::SharedPtr> mBlasObjects; ///< BLAS API objects.

        bool mRebuildBlas = true;
        Buffer::SharedPtr mpBlas;           ///< Buffer containing all BLASes.
        Buffer::SharedPtr mpBlasScratch;    ///< Scratch buffer used for BLAS builds.
    };
}
