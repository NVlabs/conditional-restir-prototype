/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
import Utils.Math.PackedFormats;
__exported import Scene.HitInfo;
__exported import Utils.Math.Ray;
__exported import Utils.Sampling.SampleGenerator;
__exported import Rendering.Materials.InteriorList;
__exported import RenderPasses.Shared.Denoising.NRDData;
__exported import PathReservoir;
import Scene.Shading;

static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

static const float kRayTMax = 1e30f;

// Be careful with changing these. PathFlags share 32-bit uint with vertexIndex. For now, we keep 10 bits for vertexIndex.
// PathFlags take higher bits, VertexIndex takes lower bits.
static const uint kVertexIndexBitCount = 10u;
static const uint kVertexIndexBitMask = (1u << kVertexIndexBitCount) - 1u;
static const uint kPathFlagsBitCount = 32u - kVertexIndexBitCount;
static const uint kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kVertexIndexBitCount;
static const uint kBSDFComponentBitOffset = kVertexIndexBitCount + 16u; // 0x10000

/** Path flags. The path flags are currently stored in kPathFlagsBitCount bits.
*/
enum class PathFlags
{
    active = 0x0001, ///< Path is active/terminated.
    hit = 0x0002,    ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission = 0x0004, ///< Scatter ray went through a transmission event.
    specular = 0x0008,     ///< Scatter ray went through a specular event.
    delta = 0x0010,        ///< Scatter ray went through a delta event.

    insideDielectricVolume = 0x0020, ///< Path vertex is inside a dielectric volume.
    lightSampledUpper = 0x0040,      ///< Last path vertex sampled lights using NEE (in upper hemisphere).
    lightSampledLower = 0x0080,      ///< Last path vertex sampled lights using NEE (in lower hemisphere).

    diffusePrimaryHit = 0x0100,         ///< Scatter ray went through a diffuse event on primary hit.
    specularPrimaryHit = 0x0200,        ///< Scatter ray went through a specular event on primary hit.
    deltaReflectionPrimaryHit = 0x0400, ///< Primary hit was sampled as the delta reflection.
    deltaTransmissionPath = 0x0800,     ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath = 0x1000,             ///< There was no non-delta events along the path so far.

    isPrevFrame = 0x2000,    
    // kUseReSTIR
    rough = 0x4000, ///< classified as a rough path vertex for ReSTIR

    ///< for ReSTIR use (for multi-component BSDF like StandardBSDF, this tells us which component is sampled)
    component1 = 0x10000, ///< e.g. component 0: DiffuseReflection component 1: DiffuseTransmission component 2: SpecularReflection
    component2 = 0x20000, ///< Component 3: SpecularReflectionTransmission
    component3 = 0x30000, /// to know the component, bit 16-17 of the path flags is checked

    nonZeroTemporalUpdate = 0x40000,

    isTraceNewSuffix = 0x80000,
    isPrefixReplay = 0x100000,
};

/** Bounce types. We keep separate counters for all of these.
*/
enum class BounceType
{
    Diffuse                 = 0,    ///< Diffuse reflection.
    Specular                = 1,    ///< Specular reflection (including delta).
    Transmission            = 2,    ///< Transmission (all kinds).
};

/** Packed path state. Contains all data from PathState in packed form.
    Default total size: 80 + (8-16) + (2*4) + (4-16) bytes.
*/
struct PackedPathState
{
    uint4 packed[5];
    PackedHitInfo hit;
    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties. The size depends on the config (4B/item).
    SampleGenerator sg;         ///< Sample generator state. Typically 4-16B.
};

struct ReSTIRGIData
{
    float3 creationPoint;       ///< Position of visible point.
    float3 creationNormal;      ///< Normal of visible point.
    float3 position;            ///< Position of secondary ray's hit point.
    float3 normal;              ///< Normal of secondary ray's hit point.
    float3 L0;                  ///< Direct lighting.
    bool   isDisabled;          ///< A flag showing if a pixel is calculated using ReSTIR GI.
};

struct ConditionalReSTIRData
{
    PathRisState pathRis;
    PathRcInfo pathRcInfo;
    float lastScatterPdf;
    float3 lastScatterWeight;
    float rrPdf;

    __init()
    {
        pathRis = PathRisState();
        pathRcInfo = {};
        lastScatterPdf = 0.f;
        lastScatterWeight = 0.f;
        rrPdf = 1.f;
    }
};

// TODO: Compact encoding to reduce live registers, e.g. packed HitInfo, packed normals.
/** Live state for the path tracer.
*/
struct PathState
{
    uint        id;                     ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.

    uint        flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                        ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    uint16_t    rejectedHits;           ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    float16_t   sceneLength;            ///< Path length in scene units (0.f at primary hit).
    uint        bounceCounters;         ///< Packed counters for different types of bounces (see BounceType).

    // Scatter ray
    float3      origin;                 ///< Origin of the scatter ray.
    float3      dir;                    ///< Scatter ray normalized direction.
    float       pdf;                    ///< Pdf for generating the scatter ray.
    float3      normal;                 ///< Shading normal at the scatter ray origin.
    HitInfo     hit;                    ///< Hit information for the scatter ray. This is populated at committed triangle hits.

    float3      thp;                    ///< Path throughput.
    float3      L;                      ///< Accumulated path contribution.

    InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties.
    SampleGenerator sg;                 ///< Sample generator state. Typically 4-16B.
    PathReplayInfo pathReplayInfo; ///< for ReSTIR
    ConditionalReSTIRData restirData;     ///< ReSTIR data (used for suffix re-generation)
    float pathTotalLength;

    // Accessors

    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); }
    bool isInsideDielectricVolume() { return hasFlag(PathFlags::insideDielectricVolume); }

    bool isRough() { return hasFlag(PathFlags::rough); }
    // TODO: generalize to arbitrary lobe types
    uint getSampledBSDFComponent() { return (flagsAndVertexIndex >> kBSDFComponentBitOffset) & 3; }
    bool isPrevFrame() { return hasFlag(PathFlags::isPrevFrame); }
    bool hasNonZeroTemporalUpdate() { return hasFlag(PathFlags::nonZeroTemporalUpdate); }
    bool isTraceNewSuffix() { return hasFlag(PathFlags::isTraceNewSuffix); }
    bool isPrefixReplay() { return hasFlag(PathFlags::isPrefixReplay); }

    bool isLightSampled()
    {
        const uint bits = (uint(PathFlags::lightSampledUpper) | uint(PathFlags::lightSampledLower)) << kVertexIndexBitCount;
        return flagsAndVertexIndex & bits;
    }

    bool isLightSampledUpper() { return hasFlag(PathFlags::lightSampledUpper); }
    bool isLightSampledLower() { return hasFlag(PathFlags::lightSampledLower); }
    bool isDiffusePrimaryHit() { return hasFlag(PathFlags::diffusePrimaryHit); }
    bool isSpecularPrimaryHit() { return hasFlag(PathFlags::specularPrimaryHit); }
    bool isDeltaReflectionPrimaryHit() { return hasFlag(PathFlags::deltaReflectionPrimaryHit); }
    bool isDeltaTransmissionPath() { return hasFlag(PathFlags::deltaTransmissionPath); }
    bool isDeltaOnlyPath() { return hasFlag(PathFlags::deltaOnlyPath); }

    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !isDelta(); }

    [mutating] void terminate() { setFlag(PathFlags::active, false); }
    [mutating] void setActive() { setFlag(PathFlags::active); }
    [mutating] void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    [mutating] void clearHit() { setFlag(PathFlags::hit, false); }

    [mutating] void clearEventFlags()
    {
        const uint bits = (uint(PathFlags::transmission) | uint(PathFlags::specular) | uint(PathFlags::delta)) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    [mutating] void setLightSampled(bool upper, bool lower) { setFlag(PathFlags::lightSampledUpper, upper); setFlag(PathFlags::lightSampledLower, lower); }
    [mutating] void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    [mutating] void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }
    [mutating] void setDeltaReflectionPrimaryHit(bool value = true) { setFlag(PathFlags::deltaReflectionPrimaryHit, value); }
    [mutating] void setDeltaTransmissionPath(bool value = true) { setFlag(PathFlags::deltaTransmissionPath, value); }
    [mutating] void setDeltaOnlyPath(bool value = true) { setFlag(PathFlags::deltaOnlyPath, value); }
    [mutating] void setRough(bool value = true) { setFlag(PathFlags::rough, value); }
    [mutating] void setSampledBSDFComponent(uint component)
    {
        flagsAndVertexIndex &= ~(3 << kBSDFComponentBitOffset);
        flagsAndVertexIndex |= component << kBSDFComponentBitOffset;
    }
    [mutating] void setPrevFrame(bool value = true) { setFlag(PathFlags::isPrevFrame, value); }
    [mutating] void setNonZeroTemporalUpdate(bool value = true) { setFlag(PathFlags::nonZeroTemporalUpdate, value); }
    [mutating] void setTraceNewSuffix(bool value = true) { setFlag(PathFlags::isTraceNewSuffix, value); }
    [mutating] void setPrefixReplay(bool value = true) { setFlag(PathFlags::isPrefixReplay, value); }

    bool hasFlag(PathFlags flag)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        return (flagsAndVertexIndex & bit) != 0;
    }

    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        flagsAndVertexIndex = (value) ? (flagsAndVertexIndex | bit) : (flagsAndVertexIndex & ~bit);
    }

    uint getBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        return (bounceCounters >> shift) & 0xff;
    }

    [mutating] void setBounces(BounceType type, uint bounces)
    {
        const uint shift = (uint)type << 3;
        bounceCounters = (bounceCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    }

    [mutating] void incrementBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        // We assume that bounce counters cannot overflow.
        bounceCounters += (1 << shift);
    }

    uint2 getPixel() { return uint2(id, id >> 12) & 0xfff; }
    uint getSampleIdx() { return id >> 24; }

    // Unsafe - assumes that index is small enough.
    [mutating] void setVertexIndex(uint index)
    {
        // Clear old vertex index.
        flagsAndVertexIndex &= kPathFlagsBitMask;
        // Set new vertex index (unsafe).
        flagsAndVertexIndex |= index;
    }

    uint getVertexIndex() { return flagsAndVertexIndex & kVertexIndexBitMask; }

    // vertex length is vertex index - 1
    int getVertexLength() { return getVertexIndex() - 1; }

    // Unsafe - assumes that vertex index never overflows.
    [mutating] void incrementVertexIndex() { flagsAndVertexIndex += 1; }
    // Unsafe - assumes that vertex index will never be decremented below zero.
    [mutating] void decrementVertexIndex() { flagsAndVertexIndex -= 1; }

    Ray getScatterRay()
    {
        return Ray(origin, dir, 0.f, kRayTMax);
    }
};
