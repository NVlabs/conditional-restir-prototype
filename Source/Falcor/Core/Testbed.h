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
#include "Window.h"
#include "RenderGraph/RenderGraph.h"
#include "Scene/Scene.h"
#include "Utils/Timing/FrameRate.h"
#include "Utils/Timing/Clock.h"
#include <memory>
#include <filesystem>

namespace Falcor
{

class ProfilerUI;

class Testbed;
using TestbedSharedPtr = std::shared_ptr<Testbed>;

/// Falcor testbed application class.
/// This is the main Falcor application available through the Python API.
class Testbed : private Window::ICallbacks
{
public:
    Testbed();
    virtual ~Testbed();

    static TestbedSharedPtr create();

    /// Run the main loop.
    /// This only returns if the application window is closed or the main loop is interrupted by calling interrupt().
    void run();

    /// Interrupt the main loop.
    void interrupt();

    /// Render a single frame.
    /// Note: This is called repeatadly when running the main loop.
    void frame();

    void loadScene(const std::filesystem::path& path);

    Scene::SharedPtr getScene() const;

    void setRenderGraph(const RenderGraph::SharedPtr& graph);
    const RenderGraph::SharedPtr& getRenderGraph() const;

private:
    // Implementation of Window::ICallbacks

    void handleWindowSizeChange() override;
    void handleRenderFrame() override;
    void handleKeyboardEvent(const KeyboardEvent& keyEvent) override;
    void handleMouseEvent(const MouseEvent& mouseEvent) override;
    void handleGamepadEvent(const GamepadEvent& gamepadEvent) override;
    void handleGamepadState(const GamepadState& gamepadState) override;
    void handleDroppedFile(const std::filesystem::path& path) override;

    void internalInit();
    void internalShutdown();

    void renderUI();

    std::shared_ptr<Window> mpWindow;
    std::shared_ptr<Fbo> mpTargetFBO;
    std::unique_ptr<Gui> mpGui;
    std::unique_ptr<ProfilerUI> mpProfilerUI;

    Scene::SharedPtr mpScene;
    RenderGraph::SharedPtr mpRenderGraph;

    FrameRate mFrameRate;
    Clock mClock;

    bool mShouldInterrupt{false};
    struct
    {
        bool showUI = true;
        bool showFPS = true;
    } mUI;
};

} // namespace Falcor
