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
#include "FalcorTest.h"
#include "Testing/UnitTest.h"
#include "RenderGraph/RenderPassLibrary.h"

#include <args.hxx>

#include <cstdio>
#include <string>
#include <vector>

FALCOR_EXPORT_D3D12_AGILITY_SDK

FalcorTest::FalcorTest(const SampleAppConfig& config, const Options& options)
    : SampleApp(config)
    , mOptions(options)
{}

FalcorTest::~FalcorTest()
{}

void FalcorTest::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    mReturnCode = runTests(std::cout, pRenderContext, getTargetFbo().get(), mOptions.filter, mOptions.xmlReportPath, mOptions.repeat);
    shutdown();
}

int main(int argc, char** argv)
{
    args::ArgumentParser parser("Falcor unit tests.");
    parser.helpParams.programName = "FalcorTest";
    args::HelpFlag helpFlag(parser, "help", "Display this help menu.", {'h', "help"});
    args::ValueFlag<std::string> deviceTypeFlag(parser, "d3d12|vulkan", "Device type.", {'d', "device-type"});
    args::ValueFlag<std::string> filterFlag(parser, "filter", "Regular expression for filtering tests to run.", {'f', "filter"});
    args::ValueFlag<std::string> xmlReportFlag(parser, "path", "XML report output file.", {'x', "xml-report"});
    args::ValueFlag<uint32_t> repeatFlag(parser, "N", "Number of times to repeat the test.", {'r', "repeat"});
    args::Flag enableDebugLayerFlag(parser, "", "Enable debug layer (enabled by default in Debug build).", {"enable-debug-layer"});
    args::CompletionFlag completionFlag(parser, {"complete"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (const args::RequiredError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    SampleAppConfig config;
    if (deviceTypeFlag)
    {
        if (args::get(deviceTypeFlag) == "d3d12")
            config.deviceDesc.type = Device::Type::D3D12;
        else if (args::get(deviceTypeFlag) == "vulkan")
            config.deviceDesc.type = Device::Type::Vulkan;
        else
        {
            std::cerr << "Invalid device type, use 'd3d12' or 'vulkan'" << std::endl;
            return 1;
        }
    }
    if (enableDebugLayerFlag) config.deviceDesc.enableDebugLayer = true;

    config.windowDesc.title = "FalcorTest";
    config.windowDesc.mode = Window::WindowMode::Minimized;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = config.windowDesc.height = 2;


    FalcorTest::Options options;
    if (filterFlag) options.filter = args::get(filterFlag);
    if (xmlReportFlag) options.xmlReportPath = args::get(xmlReportFlag);
    if (repeatFlag) options.repeat = args::get(repeatFlag);

    // Disable logging to console, we don't want to clutter the test runner output with log messages.
    Logger::setOutputs(Logger::OutputFlags::File | Logger::OutputFlags::DebugWindow);

    FalcorTest falcorTest(config, options);
    falcorTest.run();
    return falcorTest.getReturnCode();
}
