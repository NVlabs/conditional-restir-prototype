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
#include "SDFGrid.h"
#include "NormalizedDenseSDFGrid/NDSDFGrid.h"
#include "SparseVoxelSet/SDFSVS.h"
#include "SparseBrickSet/SDFSBS.h"
#include "SparseVoxelOctree/SDFSVO.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Matrix/Matrix.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <nlohmann/json.hpp>
#include <random>
#include <fstream>

using json = nlohmann::json;

namespace glm
{
    void to_json(json& j, const glm::vec3& v)
    {
        j = { v.x, v.y, v.z };
    }

    void from_json(const json& j, glm::vec3& v)
    {
        j[0].get_to(v.x);
        j[1].get_to(v.y);
        j[2].get_to(v.z);
    }
}

namespace Falcor::rmcv
{
    void to_json(json& j, const Falcor::rmcv::mat3& m)
    {
        for (uint32_t i = 0; i < 9; ++i)
        {
            j[i] = m[i / 3][i % 3];
        }
    }

    void from_json(const json& j, Falcor::rmcv::mat3& m)
    {
        for (uint32_t i = 0; i < 9; ++i)
        {
            j[i].get_to(m[i / 3][i % 3]);
        }
    }
}

namespace Falcor
{
    namespace
    {
        const std::string kEvaluateSDFPrimitivesShaderName = "Scene/SDFs/EvaluateSDFPrimitives.cs.slang";

        const char kPrimitiveShapeTypeJSONKey[] = "shape_type";
        const char kPrimitiveShapeDataJSONKey[] = "shape_data";
        const char kPrimitiveShapeBlobbingJSONKey[] = "shape_blobbing";

        const char kPrimitiveOperationTypeJSONKey[] = "operation_type";
        const char kPrimitiveOperationSmoothingJSONKey[] = "operation_smoothing";

        const char kPrimitiveTranslationJSONKey[] = "translation";
        const char kPrimitiveInvRotationScaleJSONKey[] = "inv_rot_scale";
    }

    NLOHMANN_JSON_SERIALIZE_ENUM(SDF3DShapeType, {
        { SDF3DShapeType::Sphere, "sphere" },
        { SDF3DShapeType::Ellipsoid, "ellipsoid" },
        { SDF3DShapeType::Box, "box" },
        { SDF3DShapeType::Torus, "torus" },
        { SDF3DShapeType::Cone, "cone" },
        { SDF3DShapeType::Capsule, "capsule" },
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(SDFOperationType, {
        { SDFOperationType::Union, "union" },
        { SDFOperationType::Subtraction, "subtraction" },
        { SDFOperationType::Intersection, "intersection" },
        { SDFOperationType::SmoothUnion, "smooth_union" },
        { SDFOperationType::SmoothSubtraction, "smooth_subtraction" },
        { SDFOperationType::SmoothIntersection, "smooth_intersection" },
    })

    void to_json(json& j, const SDF3DPrimitive& primitive)
    {
        j[kPrimitiveShapeTypeJSONKey] = primitive.shapeType;
        j[kPrimitiveShapeDataJSONKey] = primitive.shapeData;
        j[kPrimitiveShapeBlobbingJSONKey] = primitive.shapeBlobbing;
        j[kPrimitiveOperationTypeJSONKey] = primitive.operationType;
        j[kPrimitiveOperationSmoothingJSONKey] = primitive.operationSmoothing;
        j[kPrimitiveTranslationJSONKey] = primitive.translation;
        j[kPrimitiveInvRotationScaleJSONKey] = primitive.invRotationScale;
    }

    void from_json(const json& j, SDF3DPrimitive& primitive)
    {
        // Note: Previous serialization code stored shapeType as an uint.
        if (j[kPrimitiveShapeTypeJSONKey].is_number_unsigned())
        {
            j[kPrimitiveShapeTypeJSONKey].get_to<uint32_t>(reinterpret_cast<uint32_t&>(primitive.shapeType));
        }
        else
        {
            j[kPrimitiveShapeTypeJSONKey].get_to(primitive.shapeType);
        }
        j[kPrimitiveShapeDataJSONKey].get_to(primitive.shapeData);
        j[kPrimitiveShapeBlobbingJSONKey].get_to(primitive.shapeBlobbing);
        // Note: Previous serialization code stored operationType as an uint.
        if (j[kPrimitiveShapeTypeJSONKey].is_number_unsigned())
        {
            j[kPrimitiveOperationTypeJSONKey].get_to<uint32_t>(reinterpret_cast<uint32_t&>(primitive.operationType));
        }
        else
        {
            j[kPrimitiveOperationTypeJSONKey].get_to(primitive.operationType);
        }
        j[kPrimitiveOperationSmoothingJSONKey].get_to(primitive.operationSmoothing);
        j[kPrimitiveTranslationJSONKey].get_to(primitive.translation);
        j[kPrimitiveInvRotationScaleJSONKey].get_to(primitive.invRotationScale);
    }

    uint32_t SDFGrid::setPrimitives(const std::vector<SDF3DPrimitive>& primitives, uint32_t gridWidth)
    {
        // All types except SBS need to have a gridWidth that is a power of 2.
        Type type = getType();
        if (type != Type::SparseBrickSet)
        {
            // TODO: Expand the grid to match a grid size that is a power of 2 instead of throwing an exception.
            checkArgument(isPowerOf2(gridWidth), "'gridWidth' ({}) must be a power of 2 for SDFGrid type of {}", gridWidth, getTypeName(type));
        }

        mGridWidth = gridWidth;
        mPrimitives.clear();
        mPrimitiveIDToIndex.clear();
        mNextPrimitiveID = 0;

        return addPrimitives(primitives);
    }

    uint32_t SDFGrid::addPrimitives(const std::vector<SDF3DPrimitive>& primitives)
    {
        // Copy primitives.
        uint32_t primitivesStartOffset = (uint32_t)mPrimitives.size();
        mPrimitives.reserve(mPrimitives.size() + primitives.size());
        mPrimitives.insert(mPrimitives.end(), primitives.begin(), primitives.end());

        // Assign indirection.
        mPrimitiveIDToIndex.reserve(mPrimitives.size());
        uint32_t basePrimitiveID = mNextPrimitiveID;

        for (uint32_t idx = primitivesStartOffset; idx < mPrimitives.size(); idx++)
        {
            mPrimitiveIDToIndex[mNextPrimitiveID++] = idx;
        }

        std::unordered_set<uint32_t> indexSet;
        for (const auto& [id, index] : mPrimitiveIDToIndex)
        {
            if (!indexSet.insert(index).second)
            {
                throw RuntimeError("Multiple copies of index {}!", index);
            }
        }

        mPrimitivesDirty = true;

        updatePrimitivesBuffer();

        return basePrimitiveID;
    }

    void SDFGrid::removePrimitives(const std::vector<uint32_t>& primitiveIDs)
    {
        for (uint32_t primitiveID : primitiveIDs)
        {
            auto idxIt = mPrimitiveIDToIndex.find(primitiveID);

            if (idxIt == mPrimitiveIDToIndex.end())
            {
                logWarning("Primitive with ID {} does not exist!", primitiveID);
                continue;
            }

            // Mark as dirty.
            mPrimitivesDirty = true;

            // Baked primitives cannot be removed.
            uint32_t idx = idxIt->second;
            if (idx < mBakedPrimitiveCount)
            {
                logWarning("Primitive with ID {} has been baked, cannot remove it!", primitiveID);
                continue;
            }

            // Erase the index from the indirection map.
            mPrimitiveIDToIndex.erase(idxIt);

            // Compactify the primitive list.
            mPrimitives.erase(mPrimitives.begin() + idx);

            if (idx < mPrimitives.size())
            {
                // Update larger IDs as the primitive list must be compact.
                for (auto& [id, index] : mPrimitiveIDToIndex)
                {
                    if (index > idx)
                    {
                        --index;
                    }
                }
            }
        }

        std::unordered_set<uint32_t> indexSet;
        for (const auto& [id, index] : mPrimitiveIDToIndex)
        {
            if (!indexSet.insert(index).second)
            {
                throw RuntimeError("Multiple copies of index {}!", index);
            }
        }

        updatePrimitivesBuffer();
    }

    void SDFGrid::updatePrimitives(const std::vector<std::pair<uint32_t, SDF3DPrimitive>>& primitives)
    {
        for (auto it = primitives.begin(); it != primitives.end(); it++)
        {
            uint32_t primitiveID = it->first;
            const SDF3DPrimitive& primitive = it->second;

            auto idxIt = mPrimitiveIDToIndex.find(primitiveID);
            if (idxIt == mPrimitiveIDToIndex.end())
            {
                logWarning("Primitive with ID {} does not exist!", primitiveID);
                continue;
            }

            // Mark as dirty.
            mPrimitivesDirty = true;

            // Update the primitive.
            mPrimitives[idxIt->second] = primitive;
        }

        updatePrimitivesBuffer();
    }

    void SDFGrid::setValues(const std::vector<float>& cornerValues, uint32_t gridWidth)
    {
        // All types except SBS need to have a gridWidth that is a power of 2.
        Type type = getType();
        if (type != Type::SparseBrickSet)
        {
            checkArgument(isPowerOf2(gridWidth), "'gridWidth' ({}) must be a power of 2 for SDFGrid type of {}", gridWidth, getTypeName(type));
        }

        mGridWidth = gridWidth;

        setValuesInternal(cornerValues);
    }

    bool SDFGrid::loadValuesFromFile(const std::filesystem::path& path)
    {
        std::filesystem::path fullPath;
        if (findFileInDataDirectories(path, fullPath))
        {
            std::ifstream file(fullPath, std::ios::in | std::ios::binary);

            if (file.is_open())
            {
                uint32_t gridWidth;
                file.read(reinterpret_cast<char*>(&gridWidth), sizeof(uint32_t));

                uint32_t totalValueCount = (gridWidth + 1) * (gridWidth + 1) * (gridWidth + 1);
                std::vector<float> cornerValues(totalValueCount, 0.0f);
                file.read(reinterpret_cast<char*>(cornerValues.data()), totalValueCount * sizeof(float));

                file.close();
                setValues(cornerValues, gridWidth);

                mInitializedWithPrimitives = false;
                return true;
            }
        }

        logWarning("SDFGrid::loadValuesFromFile() file '{}' could not be opened!", path);
        return false;
    }

    void SDFGrid::generateCheeseValues(uint32_t gridWidth, uint32_t seed)
    {
        const float kHalfCheeseExtent = 0.4f;
        const uint32_t kHoleCount = 32;
        float4 holes[kHoleCount];

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (uint32_t s = 0; s < kHoleCount; s++)
        {
            float3 p = 2.0f * kHalfCheeseExtent * float3(dist(rng), dist(rng), dist(rng)) - float3(kHalfCheeseExtent);
            holes[s] = float4(p, dist(rng) * 0.2f + 0.01f);
        }

        uint32_t gridWidthInValues = 1 + gridWidth;
        uint32_t totalValueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        std::vector<float> cornerValues(totalValueCount, 0.0f);

        for (uint32_t z = 0; z < gridWidthInValues; z++)
        {
            for (uint32_t y = 0; y < gridWidthInValues; y++)
            {
                for (uint32_t x = 0; x < gridWidthInValues; x++)
                {
                    float3 pLocal = (float3(x, y, z) / float(gridWidth)) - 0.5f;
                    float sd;

                    // Create a Box.
                    {
                        float3 d = abs(pLocal) - float3(kHalfCheeseExtent);
                        float outsideDist = glm::length(float3(glm::max(d.x, 0.0f), glm::max(d.y, 0.0f), glm::max(d.z, 0.0f)));
                        float insideDist = glm::min(glm::max(glm::max(d.x, d.y), d.z), 0.0f);
                        sd = outsideDist + insideDist;
                    }

                    // Create holes.
                    for (uint32_t s = 0; s < kHoleCount; s++)
                    {
                        float4 holeData = holes[s];
                        sd = glm::max(sd, -(glm::length(pLocal - holeData.xyz) - holeData.w));
                    }

                    // We don't care about distance further away than the length of the diagonal of the unit cube where the SDF grid is defined.
                    cornerValues[x + gridWidthInValues * (y + gridWidthInValues * z)] = glm::clamp(sd, -glm::root_three<float>(), glm::root_three<float>());
                }
            }
        }

        setValues(cornerValues, gridWidth);
    }

    bool SDFGrid::writeValuesFromPrimitivesToFile(const std::filesystem::path& path, RenderContext* pRenderContext)
    {
        if (!pRenderContext) pRenderContext = gpDevice->getRenderContext();

        createEvaluatePrimitivesPass(false, mHasGridRepresentation);

        updatePrimitivesBuffer();

        uint32_t gridWidthInValues = mGridWidth + 1;
        uint32_t valueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        Buffer::SharedPtr pValuesBuffer = Buffer::createTyped<float>(valueCount);
        Buffer::SharedPtr pValuesStagingBuffer = Buffer::createTyped<float>(valueCount, Resource::BindFlags::None, Buffer::CpuAccess::Read);
        GpuFence::SharedPtr pFence = GpuFence::create();

        mpEvaluatePrimitivesPass["CB"]["gGridWidth"] = mGridWidth;
        mpEvaluatePrimitivesPass["CB"]["gPrimitiveCount"] = (uint32_t)mPrimitives.size() - mBakedPrimitiveCount;
        mpEvaluatePrimitivesPass["gPrimitives"] = mpPrimitivesBuffer;
        mpEvaluatePrimitivesPass["gOldValues"] = mHasGridRepresentation ? mpSDFGridTexture : nullptr;
        mpEvaluatePrimitivesPass["gValues"] = pValuesBuffer;
        mpEvaluatePrimitivesPass->execute(pRenderContext, uint3(gridWidthInValues));
        pRenderContext->copyResource(pValuesStagingBuffer.get(), pValuesBuffer.get());
        pRenderContext->flush(false);
        pFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        pFence->syncCpu();
        const float* pValues = reinterpret_cast<const float*>(pValuesStagingBuffer->map(Buffer::MapType::Read));

        std::ofstream file(path, std::ios::out | std::ios::binary);

        if (file.is_open())
        {
            file.write(reinterpret_cast<const char*>(&mGridWidth), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(pValues), valueCount * sizeof(float));
            file.close();
        }

        pValuesStagingBuffer->unmap();
        return true;
    }

    uint32_t SDFGrid::loadPrimitivesFromFile(const std::filesystem::path& path, uint32_t gridWidth, const std::filesystem::path& dir)
    {
        std::filesystem::path fullPath;
        if (dir.empty())
        {
            if (!findFileInDataDirectories(path, fullPath))
            {
                logWarning("File '{}' could not be found in data directories!", path);
                return 0;
            }
        }
        else
        {
            fullPath = dir / path;
        }

        std::ifstream ifs(fullPath);
        if (!ifs.good())
        {
            logWarning("Failed to open SDF grid file '{}' for reading.", fullPath);
            return false;
        }

        std::vector<SDF3DPrimitive> primitives;
        try
        {
            json j = json::parse(ifs);
            primitives = j;
        }
        catch (const std::exception& e)
        {
            logWarning("Error when deserializing SDF grid from '{}': {}", fullPath, e.what());
            return 0;
        }

        setPrimitives(primitives, gridWidth);

        mInitializedWithPrimitives = true;
        return (uint32_t)mPrimitives.size();
    }

    bool SDFGrid::writePrimitivesToFile(const std::filesystem::path& path)
    {
        std::ofstream ofs(path);
        if (!ofs.good())
        {
            logWarning("Failed to open SDF grid file '{}' for writing.", path);
            return false;
        }

        json j = mPrimitives;
        ofs << j.dump(4);

        ofs.close();

        return true;
    }

    const SDF3DPrimitive& SDFGrid::getPrimitive(uint32_t primitiveID) const
    {
        auto it = mPrimitiveIDToIndex.find(primitiveID);
        checkArgument(it != mPrimitiveIDToIndex.end(), "'primitiveID' ({}) is invalid.", primitiveID);
        return mPrimitives[it->second];
    }

    void SDFGrid::bakePrimitives(uint32_t batchSize)
    {
        // The baking is deferred, and occurs in the SDFSBS class.
        mBakedPrimitiveCount = std::min(mBakedPrimitiveCount + batchSize, (uint32_t)mPrimitives.size());

        // Tell the SDFSBS grid to bake the primitives when its update function is called.
        mBakePrimitives = true;
    }

    std::string SDFGrid::getTypeName(Type type)
    {
        switch (type)
        {
        case Type::NormalizedDenseGrid: return "NormalizedDenseGrid";
        case Type::SparseVoxelSet: return "SparseVoxelSet";
        case Type::SparseBrickSet: return "SparseBrickSet";
        case Type::SparseVoxelOctree: return "SparseVoxelOctree";
        default: FALCOR_UNREACHABLE(); return "";
        }
    }

    FALCOR_SCRIPT_BINDING(SDFGrid)
    {
        using namespace pybind11::literals;

        auto createSBS = [](const pybind11::kwargs& args)
        {
            uint32_t brickWidth = 7;
            uint32_t defaultGridWidth = 256;
            bool compressed = false;

            for (auto a : args)
            {
                auto key = a.first.cast<std::string>();
                const auto& value = a.second;

                bool isBool = pybind11::isinstance<pybind11::bool_>(value);
                bool isInt = pybind11::isinstance<pybind11::int_>(value);

                if (key == "brickWidth" && isInt)
                {
                    brickWidth = pybind11::cast<uint32_t>(value);
                }
                if (key == "defaultGridWidth" && isInt)
                {
                    defaultGridWidth = pybind11::cast<uint32_t>(value);
                }
                else if (key == "compressed" && isBool)
                {
                    compressed = pybind11::cast<bool>(value);
                }
            }
            return SDFGrid::SharedPtr(SDFSBS::create(brickWidth, compressed, defaultGridWidth));
        };

        pybind11::class_<SDFGrid, SDFGrid::SharedPtr> sdfGrid(m, "SDFGrid");
        sdfGrid.def_static("createNDGrid", [](float narrowBandThickness) { return SDFGrid::SharedPtr(NDSDFGrid::create(narrowBandThickness)); }, "narrowBandThickness"_a);
        sdfGrid.def_static("createSVS", [](){ return SDFGrid::SharedPtr(SDFSVS::create()); });
        sdfGrid.def_static("createSBS", createSBS);
        sdfGrid.def_static("createSVO", [](){ return SDFGrid::SharedPtr(SDFSVO::create()); });
        sdfGrid.def("loadValuesFromFile", &SDFGrid::loadValuesFromFile, "path"_a);
        sdfGrid.def("loadPrimitivesFromFile", &SDFGrid::loadPrimitivesFromFile, "path"_a, "gridWidth"_a, "dir"_a = "");
        sdfGrid.def("generateCheeseValues", &SDFGrid::generateCheeseValues, "gridWidth"_a, "seed"_a);
        sdfGrid.def_property("name", &SDFGrid::getName, &SDFGrid::setName);
    }

    void SDFGrid::createEvaluatePrimitivesPass(bool writeToTexture3D, bool mergeWithSDField)
    {
        if (!mpEvaluatePrimitivesPass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kEvaluateSDFPrimitivesShaderName).csEntry("main").setShaderModel("6_5");
            mpEvaluatePrimitivesPass = ComputePass::create(desc);
        }

        if (writeToTexture3D)
        {
            mpEvaluatePrimitivesPass->addDefine("_USE_SD_FIELD_3D_TEXTURE");
        }
        else
        {
            mpEvaluatePrimitivesPass->removeDefine("_USE_SD_FIELD_3D_TEXTURE");
        }

        if (mergeWithSDField)
        {
            mpEvaluatePrimitivesPass->addDefine("_MERGE_WITH_THE_SD_FIELD");
        }
        else
        {
            mpEvaluatePrimitivesPass->removeDefine("_MERGE_WITH_THE_SD_FIELD");
        }
    }

    void SDFGrid::updatePrimitivesBuffer()
    {
        if (mPrimitives.empty() || mPrimitives.size() <= mPrimitivesExcludedFromBuffer) return;

        uint32_t count = (uint32_t)mPrimitives.size() - mPrimitivesExcludedFromBuffer;
        void* pData = (void*)&mPrimitives[mPrimitivesExcludedFromBuffer];
        if (!mpPrimitivesBuffer || mpPrimitivesBuffer->getElementCount() < count)
        {
            mpPrimitivesBuffer = Buffer::createStructured(sizeof(SDF3DPrimitive), count, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, pData, false);
        }
        else
        {
            mpPrimitivesBuffer->setBlob(pData, 0, count * sizeof(SDF3DPrimitive));
        }
    }
}
