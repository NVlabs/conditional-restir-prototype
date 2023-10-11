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
#include "Testbed.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Threading.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Timing/ProfilerUI.h"
#include "Utils/UI/Gui.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include <imgui.h>

namespace Falcor
{

namespace
{
/// Global pointer holding on to the first created Testbed instance.
/// Currently, we are limited to only have one instance of the Testbed at runtime due to various global state in Falcor
/// (such as the graphics device). We also want to keep the instance alive until the end of the runtime in order to
/// allow graceful shutdown as some other objects that expect global state to still be available when shutting down.
TestbedSharedPtr spTestbed;
} // namespace

Testbed::Testbed()
{
    internalInit();
}

Testbed::~Testbed()
{
    internalShutdown();
}

TestbedSharedPtr Testbed::create()
{
    if (spTestbed)
        throw RuntimeError("Only one instance of Testbed can be created during the lifetime of the Falcor runtime.");

    spTestbed = std::make_shared<Testbed>();
    return spTestbed;
}

void Testbed::run()
{
    mShouldInterrupt = false;

    while (!mpWindow->shouldClose() && !mShouldInterrupt)
        frame();
}

void Testbed::interrupt()
{
    mShouldInterrupt = true;
}

void Testbed::frame()
{
    mClock.tick();
    mFrameRate.newFrame();

    mpWindow->pollForEvents();

    if (gpDevice)
    {
        RenderContext* pRenderContext = gpDevice->getRenderContext();

        // Clear the frame buffer.
        const float4 clearColor(1, 0, 1, 1);
        pRenderContext->clearFbo(mpTargetFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

        // Compile the render graph.
        if (mpRenderGraph)
            mpRenderGraph->compile(pRenderContext);

        // Update the scene.
        if (mpScene)
        {
            Scene::UpdateFlags sceneUpdates = mpScene->update(pRenderContext, mClock.getTime());
            if (mpRenderGraph && sceneUpdates != Scene::UpdateFlags::None)
                mpRenderGraph->onSceneUpdates(pRenderContext, sceneUpdates);
        }

        // Execute the render graph.
        if (mpRenderGraph)
        {
            (*mpRenderGraph->getPassesDictionary())[kRenderPassRefreshFlags] = RenderPassRefreshFlags::None;
            mpRenderGraph->execute(pRenderContext);

            // Blit main graph output to frame buffer.
            if (mpRenderGraph->getOutputCount() > 0)
            {
                Texture::SharedPtr pOutTex = std::dynamic_pointer_cast<Texture>(mpRenderGraph->getOutput(0));
                FALCOR_ASSERT(pOutTex);
                pRenderContext->blit(pOutTex->getSRV(), mpTargetFBO->getRenderTargetView(0));
            }
        }

        // Copy frame buffer to swap chain.
        auto pSwapChainFbo = gpDevice->getSwapChainFbo();
        pRenderContext->copyResource(pSwapChainFbo->getColorTexture(0).get(), mpTargetFBO->getColorTexture(0).get());

        renderUI();

#if FALCOR_ENABLE_PROFILER
        Profiler::instance().endFrame(pRenderContext);
#endif
        {
            FALCOR_PROFILE_CUSTOM("present", Profiler::Flags::Internal);
            gpDevice->present();
        }
    }
}

void Testbed::loadScene(const std::filesystem::path& path)
{
    mpScene = Scene::create(path);

    if (mpRenderGraph)
        mpRenderGraph->setScene(mpScene);
}

void Testbed::setRenderGraph(const RenderGraph::SharedPtr& graph)
{
    mpRenderGraph = graph;

    if (mpRenderGraph)
    {
        mpRenderGraph->onResize(mpTargetFBO.get());
        mpRenderGraph->setScene(mpScene);
    }
}

Scene::SharedPtr Testbed::getScene() const
{
    return mpScene;
}

const RenderGraph::SharedPtr& Testbed::getRenderGraph() const
{
    return mpRenderGraph;
}

// Implementation of Window::ICallbacks

void Testbed::handleWindowSizeChange()
{
    if (!gpDevice)
        return;

    // Resize the swap chain.
    uint2 winSize = mpWindow->getClientAreaSize();
    auto pSwapChainFBO = gpDevice->resizeSwapChain(winSize.x, winSize.y);

    // Resize/recreate the frame buffer.
    uint32_t width = pSwapChainFBO->getWidth();
    uint32_t height = pSwapChainFBO->getHeight();
    mpTargetFBO = Fbo::create2D(width, height, pSwapChainFBO->getDesc());

    if (mpGui)
        mpGui->onWindowResize(width, height);

    if (mpRenderGraph)
        mpRenderGraph->onResize(mpTargetFBO.get());
}

void Testbed::handleRenderFrame() {}

void Testbed::handleKeyboardEvent(const KeyboardEvent& keyEvent)
{
    if (mpGui->onKeyboardEvent(keyEvent))
        return;

    if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        if (keyEvent.key == Input::Key::Escape)
            interrupt();
        if (keyEvent.key == Input::Key::F2)
            mUI.showUI = !mUI.showUI;
        if (keyEvent.key == Input::Key::P)
            Profiler::instance().setEnabled(!Profiler::instance().isEnabled());
    }

    if (mpRenderGraph && mpRenderGraph->onKeyEvent(keyEvent))
        return;
    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return;
}

void Testbed::handleMouseEvent(const MouseEvent& mouseEvent)
{
    if (mpGui->onMouseEvent(mouseEvent))
        return;
    if (mpRenderGraph && mpRenderGraph->onMouseEvent(mouseEvent))
        return;
    if (mpScene && mpScene->onMouseEvent(mouseEvent))
        return;
}

void Testbed::handleGamepadEvent(const GamepadEvent& gamepadEvent)
{
    if (mpScene && mpScene->onGamepadEvent(gamepadEvent))
        return;
}

void Testbed::handleGamepadState(const GamepadState& gamepadState)
{
    if (mpScene && mpScene->onGamepadState(gamepadState))
        return;
}

void Testbed::handleDroppedFile(const std::filesystem::path& path) {}

// Internal

void Testbed::internalInit()
{
    OSServices::start();
    Threading::start();

    // Create the window.
    Window::Desc windowDesc;
    windowDesc.title = "Testbed";
    mpWindow = Window::create(windowDesc, this);

#if FALCOR_WINDOWS
    // Set the icon.
    setWindowIcon("framework/nvidia.ico", mpWindow->getApiHandle());
#endif

    // Create the device.
    Device::Desc deviceDesc;
    gpDevice = Device::create(mpWindow, deviceDesc);

    // Set global shader defines.
    Program::DefineList globalDefines = {
        {"FALCOR_NVAPI_AVAILABLE", (FALCOR_NVAPI_AVAILABLE && gpDevice->getType() == Device::Type::D3D12) ? "1" : "0"},
    };
    Program::addGlobalDefines(globalDefines);

    // Create the frame buffer.
    auto pSwapChainFbo = gpDevice->getSwapChainFbo();
    mpTargetFBO = Fbo::create2D(pSwapChainFbo->getWidth(), pSwapChainFbo->getHeight(), pSwapChainFbo->getDesc());

    // Create the GUI.
    mpGui = Gui::create(pSwapChainFbo->getWidth(), pSwapChainFbo->getHeight(), getDisplayScaleFactor());

    mFrameRate.reset();

    handleWindowSizeChange();
}

void Testbed::internalShutdown()
{
    mpRenderGraph.reset();
    mpScene.reset();

    if (gpDevice)
        gpDevice->flushAndSync();

    Clock::shutdown();
    Threading::shutdown();

    mpGui.reset();
    mpTargetFBO.reset();

    gpDevice->cleanup();
    gpDevice.reset();

    OSServices::stop();
}

void Testbed::renderUI()
{
    FALCOR_PROFILE("renderUI");

    mpGui->beginFrame();

    // Help screen.
    {
        if (!ImGui::IsPopupOpen("##Help") && ImGui::IsKeyPressed(ImGuiKey_F1))
            ImGui::OpenPopup("##Help");

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(50, 50));
        if (ImGui::BeginPopupModal(
                "##Help", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration
            ))
        {
            ImGui::Text(
                "Help\n"
                "\n"
                "ESC - Exit (or return to Python interpreter)\n"
                "F1  - Show this help screen\n"
                "F2  - Show/hide UI\n"
                "P   - Enable/disable profiler\n"
                "\n"
            );

            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape))
                ImGui::CloseCurrentPopup();

            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
    }

    if (mUI.showUI)
    {
        // FPS display.
        if (mUI.showFPS)
        {
            Gui::Window w(
                mpGui.get(), "##FPS", {0, 0}, {10, 10},
                Gui::WindowFlags::AllowMove | Gui::WindowFlags::AutoResize | Gui::WindowFlags::SetFocus
            );
            w.text(mFrameRate.getMsg());
        }

        // Profiler.
        {
            auto& profiler = Profiler::instance();

            if (profiler.isEnabled())
            {
                bool open = profiler.isEnabled();
                Gui::Window profilerWindow(mpGui.get(), "Profiler", open, {800, 350}, {10, 10});
                profiler.endEvent("renderUI"); // Suspend renderUI profiler event

                if (open)
                {
                    if (!mpProfilerUI)
                        mpProfilerUI = ProfilerUI::create(Profiler::instancePtr());

                    mpProfilerUI->render();
                    profiler.startEvent("renderUI");
                    profilerWindow.release();
                }

                profiler.setEnabled(open);
            }
        }

        {
            Gui::Window w(mpGui.get(), "Render Graph", {300, 300}, {10, 50});
            if (mpRenderGraph)
                mpRenderGraph->renderUI(w);
            else
                w.text("No render graph loaded");
        }

        {
            Gui::Window w(mpGui.get(), "Scene", {300, 300}, {10, 360});
            if (mpScene)
                mpScene->renderUI(w);
            else
                w.text("No scene loaded");
        }
    }

    mpGui->render(gpDevice->getRenderContext(), gpDevice->getSwapChainFbo(), (float)mFrameRate.getLastFrameTime());
}

FALCOR_SCRIPT_BINDING(Testbed)
{
    using namespace pybind11::literals;

    pybind11::class_<Testbed, std::shared_ptr<Testbed>> testbed(m, "Testbed");
    testbed.def(pybind11::init(&Testbed::create));
    testbed.def("run", &Testbed::run);
    testbed.def("frame", &Testbed::frame);
    testbed.def("loadScene", &Testbed::loadScene, "path"_a);

    testbed.def_property_readonly("scene", &Testbed::getScene);
    testbed.def_property("renderGraph", &Testbed::getRenderGraph, &Testbed::setRenderGraph);
}

} // namespace Falcor
