add_renderpass(PathTracer)

target_sources(PathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    NRDHelpers.slang
    ConditionalReSTIRData.slang
    Params.slang
    PathState.slang
    PathTracer.slang
    PathTracer.cpp
    PathTracer.h
    PathTracerNRD.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.cs.slang
    UserInteractionRecorder.h
)

target_copy_shaders(PathTracer RenderPasses/PathTracer)

target_source_group(PathTracer "RenderPasses")
