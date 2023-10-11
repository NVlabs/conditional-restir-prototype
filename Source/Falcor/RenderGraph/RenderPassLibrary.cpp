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
#include "RenderPassLibrary.h"
#include "RenderGraph.h"
#include "RenderPasses/ResolvePass.h"
#include "Core/API/Device.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <fstream>

namespace Falcor
{
    static const std::string kDllSuffix = ".falcor";

    template<typename Pass>
    using PassFunc = typename Pass::SharedPtr(*)(RenderContext* pRenderContext, const Dictionary&);

    RenderPassLibrary& RenderPassLibrary::instance()
    {
        static RenderPassLibrary sInstance; // TODO: REMOVEGLOBAL
        return sInstance;
    }

    RenderPassLibrary::RenderPassLibrary()
    {
        // Add builtin passes.
        registerPass(ResolvePass::kInfo, ResolvePass::create);
    }

    RenderPassLibrary::~RenderPassLibrary()
    {
        releaseLibraries();
    }

    void RenderPassLibrary::releaseLibraries()
    {
        mPasses.clear();
        while (!mLibs.empty())
            releaseLibrary(mLibs.begin()->first);
    }

    RenderPassLibrary& RenderPassLibrary::registerPass(const RenderPass::Info& info, CreateFunc func)
    {
        registerInternal(info, func, nullptr);
        return *this;
    }

    void RenderPassLibrary::registerInternal(const RenderPass::Info& info, CreateFunc func, SharedLibraryHandle library)
    {
        if (mPasses.find(info.type) != mPasses.end())
        {
            logWarning("Trying to register a render-pass '{}' to the render-passes library, but a render-pass with the same name already exists. Ignoring the new definition.", info.type);
        }
        else
        {
            mPasses[info.type] = ExtendedDesc(info, func, library);
        }
    }

    std::shared_ptr<RenderPass> RenderPassLibrary::createPass(RenderContext* pRenderContext, const std::string& type, const Dictionary& dict)
    {
        if (mPasses.find(type) == mPasses.end())
        {
            // See if we can load a DLL with the render passes's type name and retry
            logInfo("Can't find a render-pass named '{}'. Trying to load a render-pass library '{}'.", type, type);
            loadLibrary(type);

            if (mPasses.find(type) == mPasses.end())
            {
                logWarning("Trying to create a render-pass named '{}', but no such type exists in the library.", type);
                return nullptr;
            }
        }

        auto& renderPass = mPasses[type];
        return renderPass.func(pRenderContext, dict);
    }

    RenderPassLibrary::DescVec RenderPassLibrary::enumerateClasses() const
    {
        DescVec v;
        v.reserve(mPasses.size());
        for (const auto& p : mPasses) v.push_back(p.second);
        return v;
    }

    std::vector<std::string> RenderPassLibrary::enumerateLibraries()
    {
        std::vector<std::string> result;
        for (const auto& lib : mLibs)
        {
            result.push_back(lib.first);
        }
        return result;
    }

    void RenderPassLibrary::loadLibrary(const std::string& filename)
    {
        auto path = getRuntimeDirectory() / filename;
#if FALCOR_WINDOWS
        path.replace_extension(".dll");
#elif FALCOR_LINUX
        path.replace_extension(".so");
#endif

        if (!std::filesystem::exists(path))
        {
            logWarning("Can't load render-pass library '{}'. File not found.", path);
            return;
        }

        if (mLibs.find(filename) != mLibs.end())
        {
            logInfo("Render-pass library '{}' already loaded. Ignoring 'loadLibrary()' call.", filename);
            return;
        }

        SharedLibraryHandle library = loadSharedLibrary(path);
        if (library == nullptr)
        {
            logWarning("Can't load render-pass library '{}'. File is not a shared library.", path);
            return;
        }

        auto getPassesProc = (LibraryFunc)getProcAddress(library, "getPasses");
        if (getPassesProc == nullptr)
        {
            logWarning("Can't load render-pass library '{}'. Library does not export a 'getPasses' procedure.", path);
            return;
        }

        mLibs[filename] = library;
        getPassesProc(*this);

        // Re-import falcor package to current (executing) scripting context.
        auto ctx = Scripting::getCurrentContext();
        if (Scripting::isRunning()) Scripting::runScript("from falcor import *", ctx);
    }

    void RenderPassLibrary::releaseLibrary(const std::string& filename)
    {
        auto libIt = mLibs.find(filename);
        if (libIt == mLibs.end())
        {
            logWarning("Can't unload render-pass library '{}'. The library wasn't loaded.", filename);
            return;
        }

        if (gpDevice)
        {
            gpDevice->flushAndSync();
        }

        // Delete all the classes that were owned by the library
        SharedLibraryHandle library = libIt->second;
        for (auto it = mPasses.begin(); it != mPasses.end();)
        {
            if (it->second.library == library) it = mPasses.erase(it);
            else ++it;
        }

        releaseSharedLibrary(library);
        mLibs.erase(libIt);
    }
}
