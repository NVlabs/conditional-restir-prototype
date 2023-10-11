/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/

import Utils.Sampling.SampleGenerator;
__exported import Scene.HitInfo;
import Utils.Debug.PixelDebug;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;

struct PrefixInfo
{
    float3 thp;
}

struct PrefixReservoir
{
    uint McomponentType;
    float W; // stores UCW for prefix resampling
    float rcJacobian;
    float pHat;
    float pathFootprint;

    __init()
    {
        McomponentType = 1 << 2;
        W = 1.f;
        rcJacobian = 0.f;
        pHat = 0.f;
        pathFootprint = 0.f;
    }

    __init(uint _componentType, bool needRandomReplay, float _rcJacobian, float _pHat, float _pathFootprint, float _W = 1.f, int M = 1)
    {
        McomponentType = _componentType;
        W = _W;
        McomponentType = (M << 3) | (int(needRandomReplay) << 2) | _componentType;
        rcJacobian = _rcJacobian;
        pHat = _pHat;
        pathFootprint = _pathFootprint;
    }

    int M()
    {
        return McomponentType >> 3;
    }

    [mutating] void setM(int M)
    {
        McomponentType &= 0x7;
        McomponentType |= M << 3;
    }

    [mutating] void increaseM(int inc)
    {
        McomponentType += inc << 3;
    }

    [mutating] void setNeedRandomReplay(bool val)
    {
        McomponentType &= ~0x4;
        McomponentType |= int(val) << 2;
    }

    bool needRandomReplay()
    {
        return (McomponentType >> 2) & 1;
    }

    uint componentType()
    {
        return McomponentType & 0x3;
    }

    [mutating] void setComponentType(uint type)
    {
        McomponentType &= ~0x3;
        McomponentType |= type;
    }
}

struct PrefixSearchKey
{
#if COMPRESS_PREFIX_SEARCH_ENTRY
    uint packedWo;
    property float3 wo
    {
        get { return decodeNormal2x16(packedWo); }
        set { packedWo = encodeNormal2x16(newValue); }
    }
#else    
    float3 wo;
#endif
}

struct PackedBoundingBox
{
    float3 minPoint;
    float3 maxPoint;
    float Pad0;
    float Pad1;
};

struct PrefixGBuffer
{
    HitInfo hit;

#if !USE_RESERVOIR_COMPRESSION
    float3 wo;
#else
    uint packedWo;
    property float3 wo
    {
        get { return decodeNormal2x16(packedWo); }
        set { packedWo = encodeNormal2x16(newValue); }
    }
#endif

    __init()
    {
        hit = {};
        wo = 0.f;
    }


    __init(HitInfo _hit, float3 _wo)
    
    {
        hit = _hit;
        wo = _wo;
    }
}

struct LastVertexState
{
    uint data = 0;

    __init(uint _data)
    {
        data = _data;
    }

    /// WARNING: change mask in two getLastVertexState()s when adding a new field!!! 

    [mutating] void Init(bool isCurrentVertexFarFromPrev, uint lastBSDFComponent,
              bool isLastVertexDelta, bool isLastVertexTransmission, bool isLastVertexRough, bool isVertexClassifiedAsRoughForNEE)
    {
        data = 0;
        data |= (uint)isLastVertexDelta;
        data |= (uint)isLastVertexTransmission << 1;
        data |= (uint)lastBSDFComponent << 2;
        data |= (uint)isLastVertexRough << 4;
        data |= (uint)isCurrentVertexFarFromPrev << 5;
        data |= (uint)isVertexClassifiedAsRoughForNEE << 6;
    }

    bool isLastVertexDelta()
    {
        return data & 1;
    }

    bool isLastVertexTransmission()
    {
        return (data >> 1) & 1;
    }

    uint lastBSDFComponent()
    {
        return (data >> 2) & 3;
    }

    bool isLastVertexRough()
    {
        return (data >> 4) & 1;
    }

    bool isCurrentVertexFarFromPrev()
    {
        return (data >> 5) & 1;
    }

    bool isVertexClassifiedAsRoughForNEE()
    {
        return (data >> 6) & 1;
    }
}

struct ReSTIRPathFlags
{
    uint flags = 0;

    // assuming data contains from bit 0 to bit 3, isDelta, isTransmission, and BSDFComponent
    [mutating] void tryInsertBounceTypesBeforeRcVertex(uint data, bool doInsert)
    {
        flags &= doInsert ? ~(0xf00000) : 0xffffffff;
        flags |= doInsert ? (data << 20) : 0;
    }

    [mutating] void tryInsertBounceTypesAfterRcVertex(uint data, bool doInsert)
    {
        flags &= doInsert ? ~(0xf000000) : 0xffffffff;
        flags |= doInsert ? (data << 24) : 0;
    }

    [mutating]
    void transferDeltaTransmissionBSDFEvent(ReSTIRPathFlags other, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= other.flags & (beforeRcVertex ? 0xf00000 : 0xf000000);//  << (beforeRcVertex ? 20 : 24);
    }

    [mutating]
    void transferDeltaTransmissionBSDFEvent(uint pathFlagData, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= pathFlagData & (beforeRcVertex ? 0xf00000 : 0xf000000); //  << (beforeRcVertex ? 20 : 24);
    }

    [mutating]
    void transferDeltaTransmissionBSDFEvent(LastVertexState lvs, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= (lvs.data << (beforeRcVertex ? 20 : 24)) & (beforeRcVertex ? 0xf00000 : 0xf000000); //  << (beforeRcVertex ? 20 : 24);
    }

    [mutating]
    void insertDeltaTransmissionBSDFEvent(bool isDeltaEvent, bool isTransmissionEvent, uint component, bool beforeRcVertex)
    {
        uint data = uint(isDeltaEvent) | uint(isTransmissionEvent) << 1 | component << 2;
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= data << (beforeRcVertex ? 20 : 24);
    }

    [mutating]
    void insertIsDeltaEvent(bool isDeltaEvent, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0x100000) : ~(0x1000000));
        flags |= uint(isDeltaEvent) << (beforeRcVertex ? 20 : 24);
    }

    [mutating]
    void insertIsTransmissionEvent(bool isTransmissionEvent, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0x200000) : ~(0x2000000));
        flags |= uint(isTransmissionEvent) << (beforeRcVertex ? 21 : 25);
    }

    // component uses 2 bits
    [mutating]
    void insertBSDFComponentType(uint component, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xc00000) : ~(0xc000000));
        flags |= component << (beforeRcVertex ? 22 : 26);
    }

    bool decodeIsDeltaEvent(bool beforeRcVertex)
    {
        return (flags >> (beforeRcVertex ? 20 : 24)) & 1;
    }

    bool decodeIsTransmissionEvent(bool beforeRcVertex)
    {
        return (flags >> (beforeRcVertex ? 21 : 25)) & 1;
    }

    uint decodeBSDFComponent(bool beforeRcVertex)
    {
        return (flags >> (beforeRcVertex ? 22 : 26)) & 3;
    }

    // maximum length: 15
    [mutating]
    void insertPathLength(int pathLength)
    {
        flags &= ~0xF;
        flags |= pathLength & 0xF;
    }

    // maximum length: 15
    [mutating]
    void insertRcVertexLength(int rcVertexLength)
    {
        flags &= ~0xF0;
        flags |= (rcVertexLength & 0xF) << 4;
    }

    // [mutating]
    // void insertSuffixRcVertexLength(int suffixRcVertexLength)
    // {
    //     flags &= ~0xF00;
    //     flags |= (suffixRcVertexLength & 0xF) << 8;
    // }

    [mutating]
    void insertPathTreeLength(int length)
    {
        flags &= ~0xF000;
        flags |= (length & 0xF) << 12;
    }

    [mutating]
    void insertPrefixLength(int length)
    {
        flags &= 0x0FFFFFFF;
        flags |= (length & 0xF) << 28;
    }

    int pathLength()
    {
        return flags & 0xF;
    }

    int prefixLength()
    {
        return (flags >> 28) & 0xF;
    }

    int rcVertexLength()
    {
        return (flags >> 4) & 0xF;
    }

    int pathTreeLength()
    {
        return (flags >> 12) & 0xF;
    }

    [mutating]
    void insertLastVertexNEE(bool isNEE)
    {
        flags &= ~0x10000;
        flags |= (int(isNEE) & 1) << 16;
    }

    bool lastVertexNEE()
    {
        return (flags >> 16) & 1;
    }

    // reserve bit 16
    [mutating]
    void insertLightType(uint lightType)
    {
        flags &= ~0xc0000;
        flags |= ((int(lightType) & 3) << 18);
    }

    uint lightType()
    {
        return (flags >> 18) & 3;
    }

    [mutating]
    void insertUserFlag(bool val)
    {
        flags &= ~0x20000;
        flags |= (int(val) & 1) << 17;
    }

    bool isUserFlagSet()
    {
        return (flags >> 17) & 1;
    }
};

struct PathReservoir
{
    float M;
    float weight;
    float3 integrand; // the integrand value f/p in PSS
    ReSTIRPathFlags pathFlags;
    uint initRandomSeed;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE    
    uint rcRandomSeed;
#endif
    uint suffixInitRandomSeed;
    float lightPdf; // only useful if the path ends at rcVertex

    HitInfo rcHit;

    float rcJacobian;
    float3 rcIrrad; // throughput (might only store throughput after rc)

#if !USE_RESERVOIR_COMPRESSION
    float3 rcWi;    //
#else
    uint packedRcWi;    //
    property float3 rcWi
    {
        get { return decodeNormal2x16(packedRcWi); }
        set { packedRcWi = encodeNormal2x16(newValue); }
    }
#endif

    // __init(PathReservoirEssential _data)
    // {
    // }

    [mutating]
    void init()
    {
        this.M = 0;
        this.integrand = 0;
        this.weight = 0.;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = 0;
#endif        
        this.initRandomSeed = 0;
        this.suffixInitRandomSeed = 0;
        this.pathFlags.flags = 0;
    }

    [mutating] void transferRisStateNoInitRandomSeed(const PathRisState state, uint suffixInitRandomSeed)
    {
        this.M = 1;
        this.weight = state.weight;
        this.integrand = state.integrand;
        this.lightPdf = state.lightPdf;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = state.rcRandomSeed;
#endif        
        //this.initRandomSeed = initRandomSeed;
        this.suffixInitRandomSeed = suffixInitRandomSeed;
        this.rcHit = state.rcHit;
        this.pathFlags = state.pathFlags;
        this.rcIrrad = state.rcIrrad;
        this.rcJacobian = state.rcJacobian;
        this.rcWi = state.rcWi;
    }

    static PathReservoir createFromRisState(const PathRisState state, uint initRandomSeed, uint suffixInitRandomSeed)
    {
        PathReservoir r;
        r.M = 1;
        r.weight = state.weight;
        r.integrand = state.integrand;
        r.lightPdf = state.lightPdf;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        r.rcRandomSeed = state.rcRandomSeed;
#endif        
        r.initRandomSeed = initRandomSeed;
        r.suffixInitRandomSeed = suffixInitRandomSeed;
        r.rcHit = state.rcHit;
        r.pathFlags = state.pathFlags;
        r.rcIrrad = state.rcIrrad;
        r.rcJacobian = state.rcJacobian;
        r.rcWi = state.rcWi;
        return r;
    }

    // TODO: this function should accept reservoir as input
    [mutating]
    bool add(const PathRisState state, uint initRandomSeed, uint suffixInitRandomSeed, inout SampleGenerator sg)
    {
        float w = state.weight;
        w = (isnan(w) || isinf(w)) ? 0.0f : w;
        this.M += 1;
        this.weight += w;

        // Accept?
        bool sampleSelected = (sampleNext1D(sg) * this.weight < w);

        this.pathFlags.flags = sampleSelected ? state.pathFlags.flags : this.pathFlags.flags;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = sampleSelected ? state.rcRandomSeed : this.rcRandomSeed;
#endif        
        this.initRandomSeed = sampleSelected ? initRandomSeed : this.initRandomSeed;
        this.suffixInitRandomSeed = sampleSelected ? suffixInitRandomSeed : this.suffixInitRandomSeed;
        this.lightPdf = sampleSelected ? state.lightPdf : this.lightPdf;
        this.integrand = sampleSelected ? state.integrand : this.integrand;
        this.rcIrrad = sampleSelected ? state.rcIrrad : this.rcIrrad;
        this.rcJacobian = sampleSelected ? state.rcJacobian : this.rcJacobian;
        this.rcWi = sampleSelected ? state.rcWi : this.rcWi;

        this.rcHit.data = sampleSelected ? state.rcHit.data : this.rcHit.data;


        return sampleSelected;
    }


    [mutating]
    bool mergeSameDomain(PathReservoir inReservoir, inout SampleGenerator sg, float misWeight)
    {
        float w = PathRisState::toScalar(inReservoir.integrand) * inReservoir.weight * misWeight;
        w = (isnan(w) || isinf(w)) ? 0.0f : w;

        this.M += inReservoir.M;
        this.weight += w;

        // Accept?
        bool sampleSelected = (sampleNext1D(sg) * this.weight < w);

        this.pathFlags.flags = sampleSelected ? inReservoir.pathFlags.flags : this.pathFlags.flags;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = sampleSelected ? inReservoir.rcRandomSeed : this.rcRandomSeed;
#endif        
        this.initRandomSeed = sampleSelected ? inReservoir.initRandomSeed : this.initRandomSeed;
        this.suffixInitRandomSeed = sampleSelected ? inReservoir.suffixInitRandomSeed : this.suffixInitRandomSeed;
        this.lightPdf = sampleSelected ? inReservoir.lightPdf : this.lightPdf;

        this.rcHit.data = sampleSelected ? inReservoir.rcHit.data : this.rcHit.data;

        this.integrand = sampleSelected ? inReservoir.integrand : this.integrand;
        this.rcJacobian = sampleSelected ? inReservoir.rcJacobian : this.rcJacobian;
        this.rcIrrad = sampleSelected ? inReservoir.rcIrrad : this.rcIrrad;
#if !USE_RESERVOIR_COMPRESSION
        this.rcWi = sampleSelected ? inReservoir.rcWi : this.rcWi;
#else
        this.packedRcWi = sampleSelected ? inReservoir.packedRcWi : this.packedRcWi;
#endif

        return sampleSelected;
    }

    [mutating]
    bool merge(float3 integrand, float in_Jacobian, PathReservoir inReservoir, inout SampleGenerator sg, float misWeight)
    {
        float w = PathRisState::toScalar(integrand) * in_Jacobian * inReservoir.weight * misWeight;
        w = (isnan(w) || isinf(w)) ? 0.0f : w;

        this.M += inReservoir.M;
        this.weight += w;

        // Accept?
        bool sampleSelected = (sampleNext1D(sg) * this.weight < w);

        this.pathFlags.flags = sampleSelected ? inReservoir.pathFlags.flags : this.pathFlags.flags;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = sampleSelected ? inReservoir.rcRandomSeed : this.rcRandomSeed;
#endif        
        this.initRandomSeed = sampleSelected ? inReservoir.initRandomSeed : this.initRandomSeed;
        this.suffixInitRandomSeed = sampleSelected ? inReservoir.suffixInitRandomSeed : this.suffixInitRandomSeed;
        this.lightPdf = sampleSelected ? inReservoir.lightPdf : this.lightPdf;

        this.rcHit.data = sampleSelected ? inReservoir.rcHit.data : this.rcHit.data;

        this.integrand = sampleSelected ? integrand : this.integrand;
        this.rcJacobian = sampleSelected ? inReservoir.rcJacobian : this.rcJacobian;
        this.rcIrrad = sampleSelected ? inReservoir.rcIrrad : this.rcIrrad;
#if !USE_RESERVOIR_COMPRESSION
        this.rcWi = sampleSelected ? inReservoir.rcWi : this.rcWi;
#else
        this.packedRcWi = sampleSelected ? inReservoir.packedRcWi : this.packedRcWi;
#endif

        return sampleSelected;
    }

    // multiply W with pHat
    [mutating]
    void beginGRIS(float selfMISWeight)
    {
        float p_hat = PathRisState::toScalar(integrand);
        this.weight *= p_hat * selfMISWeight;
    }

    // assuming using proper resampling MIS weight, no need to divide by M
    [mutating]
    void finalizeGRIS()
    {
        float p_hat = PathRisState::toScalar(integrand);
        this.weight = (p_hat == 0.f) ? 0.f : (this.weight / p_hat);
    }

    [mutating]
    void finalizeGRISDivideByM() // this is for more traditional streaming style
    {
        float p_hat = PathRisState::toScalar(integrand);
        this.weight = (p_hat == 0.f) ? 0.f : (this.weight / (this.M * p_hat));
    }

    // call this if M needs to be reset to 1 (usually used outside spatiotemporal reuse)
    [mutating]
    void resetM()
    {
        this.M = 1;
    }

    [mutating]
    void packPartialRcJacobianInInitRandomSeed(float jacobian)
    {
        this.initRandomSeed = asuint(jacobian);
    }

    float getPartialRcJacobianFromInitRandomSeed()
    {
        return asfloat(this.initRandomSeed);
    }
}

// stores information about reconnection vertex (that can be reused by later vertices)
struct PathReplayInfo
{
    ReSTIRPathFlags pathFlags;
    float3 rcThroughput; // path throughput after rcVertex
    
    HitInfo rcHit;
    
    float rcJacobian;
    uint suffixInitRandomSeed;
#if !USE_RESERVOIR_COMPRESSION
    float3 rcWi;
#else
    uint packedRcWi;
    property float3 rcWi
    {
        get { return decodeNormal2x16(packedRcWi); }
        set { packedRcWi = encodeNormal2x16(newValue); }
    }
#endif
}

struct PathRcInfo
{
    // also stores random replay path information
    ReSTIRPathFlags pathFlags; ///< for ReSTIR
    uint rcRandomSeed;         // to recover NEE information
    uint initRandomSeed;
    uint suffixInitRandomSeed;
    // also stores outgoing direction at the previous vertex of rcVertex, used in hybrid shift replay
    float3 rcThroughput; // path throughput after rcVertex
    // also stores the hit for the previous vertex of rcVertex, used in hybrid shift replay

    HitInfo rcHit;

    float rcJacobian;
#if !USE_RESERVOIR_COMPRESSION
    float3 rcWi; // can be computed outside
#else
    uint packedRcWi; // can be computed outside

    property float3 rcWi
    {
        get { return decodeNormal2x16(packedRcWi); }
        set { packedRcWi = encodeNormal2x16(newValue); }
    }
#endif
}

// two of these?
struct PathRisState
{
    float weight;
    float3 integrand; // the integrand value f/p in PSS
    ReSTIRPathFlags pathFlags;
    float lightPdf; // only useful if the path ends at rcVertex
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
    uint rcRandomSeed;
#endif    

    HitInfo rcHit;

    float rcJacobian;
    float3 rcIrrad; // throughput (might only store throughput after rc)
#if !USE_RESERVOIR_COMPRESSION
    float3 rcWi;    //
#else
    uint packedRcWi;    //
    property float3 rcWi
    {
        get { return decodeNormal2x16(packedRcWi); }
        set { packedRcWi = encodeNormal2x16(newValue); }
    }
#endif

    __init()
    {
        pathFlags.flags = 0;
        pathFlags.insertPathLength(0);
        pathFlags.insertPathTreeLength(0);
        pathFlags.insertRcVertexLength(15);
        pathFlags.insertPrefixLength(0);
        rcIrrad = 0.f;
        integrand = 0.f;
        weight = 0.f;
        lightPdf = 0.f;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        rcRandomSeed = 0;
#endif        
        rcWi = 0.f;
        rcJacobian = 0.f;
        rcHit = {};
    }

    [mutating]
    bool addBSDFVertex(inout SampleGenerator sg, PathRcInfo pathRcInfo, uint pathLength,
                       float3 pathContribution, float3 wi, float3 Le, float3 Lr, float russianRoulettePDF, float lightPdf, uint lightType)
    {
        bool selected = add(sg, pathContribution, russianRoulettePDF);
        //print(selected ? "---accepted" : "--rejected", 1111);
        uint rcVertexLength = pathRcInfo.pathFlags.rcVertexLength();
        bool isRcVertex = pathLength == rcVertexLength;
        // rcVertexLength is the same as in pathRcInfo.pathFlags

        this.pathFlags.flags = selected ? pathRcInfo.pathFlags.flags : this.pathFlags.flags;
        this.pathFlags.insertPathLength(selected ? pathLength : this.pathFlags.pathLength());
        this.pathFlags.insertLastVertexNEE(!selected && this.pathFlags.lastVertexNEE());
        this.rcIrrad = selected ? (isRcVertex ? Le : (pathRcInfo.rcThroughput * Lr)) : this.rcIrrad;

        this.rcWi = selected ? pathRcInfo.rcWi : this.rcWi;
        if (selected) this.rcHit = pathRcInfo.rcHit;
        this.rcJacobian = selected ? pathRcInfo.rcJacobian : this.rcJacobian;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = selected ? pathRcInfo.rcRandomSeed : this.rcRandomSeed;
#endif
        this.integrand = selected ? pathContribution : this.integrand;
        this.lightPdf = selected ? lightPdf : this.lightPdf;
        this.pathFlags.insertLightType(selected ? lightType : this.pathFlags.lightType());

        return selected;
    }

    [mutating]
    bool addNeeVertex(inout SampleGenerator sg, PathRcInfo pathRcInfo, bool isRcVertex, uint pathLength,
                      float3 pathContribution, float3 wi, float3 Le, float3 Lr, float russianRoulettePDF, float lightPdf, uint lightType)
    {
        bool selected = add(sg, pathContribution, russianRoulettePDF);
        uint rcVertexLength = pathRcInfo.pathFlags.rcVertexLength();
        rcVertexLength = isRcVertex ? pathLength : rcVertexLength;
        this.pathFlags.flags = selected ? pathRcInfo.pathFlags.flags : this.pathFlags.flags;
        this.pathFlags.insertPathLength(selected ? pathLength : this.pathFlags.pathLength());
        this.pathFlags.insertRcVertexLength(selected ? rcVertexLength : this.pathFlags.rcVertexLength());
        this.pathFlags.insertLastVertexNEE(selected || this.pathFlags.lastVertexNEE());
        this.rcIrrad = selected ? (isRcVertex ? Le : pathRcInfo.rcThroughput * Lr) : this.rcIrrad;
        this.rcWi = selected ? (isRcVertex ? wi : pathRcInfo.rcWi) : this.rcWi;
        // will be overwritten for isRcVertex
        if (selected) this.rcHit = pathRcInfo.rcHit;
        this.rcJacobian = selected ? pathRcInfo.rcJacobian : this.rcJacobian;
#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
        this.rcRandomSeed = selected ? pathRcInfo.rcRandomSeed : this.rcRandomSeed;
#endif        
        this.integrand = selected ? pathContribution : this.integrand;
        this.lightPdf = selected ? lightPdf : this.lightPdf;
        this.pathFlags.insertLightType(selected ? lightType : this.pathFlags.lightType());

        return selected;
    }

    [mutating]
    void markAsRcVertex(uint pathLength,

        HitInfo hit,

        LastVertexState lastVertexState,
        float lightPdf, uint lightType, float3 Le, float3 wi, float rcJacobian)
    {
        this.pathFlags.insertRcVertexLength(pathLength);

        this.rcHit = hit;
        this.pathFlags.insertIsDeltaEvent(lastVertexState.isLastVertexDelta(), true);
        this.pathFlags.insertIsTransmissionEvent(lastVertexState.isLastVertexTransmission(), true);
        this.pathFlags.insertBSDFComponentType(lastVertexState.lastBSDFComponent(), true);
        this.lightPdf = lightPdf;

        this.rcJacobian = rcJacobian;

        this.pathFlags.insertLightType(lightType);
        this.rcIrrad = Le;
        this.rcWi = wi;
    }

    [mutating]
    bool add(inout SampleGenerator sg, float3 pathContribution, float russianRoulettePDF)
    {
        float w = toScalar(pathContribution) / russianRoulettePDF;

        w = isnan(w) ? 0.0f : w;
        this.weight += w;

        return (sampleNext1D(sg) * this.weight < w);
    }

    static float toScalar(float3 color)
    {
        return luminance(color); // luminance
    }
}

// each store a pixelID+pixelOffset (or two pixel offsets)
// to produce key, we need a path length, and screen region in morton code order
struct ReconnectionData
{
    HitInfo rcPrevHit; 

    float3 pathThroughput;

#if !USE_RESERVOIR_COMPRESSION
    float3 rcPrevWo;      
#else
    uint packedRcPrevWo; 
    property float3 rcPrevWo
    {
        get { return decodeNormal2x16(packedRcPrevWo); }
        set { packedRcPrevWo = encodeNormal2x16(newValue); }
    }
#endif


    __init(HitInfo rcPrevHit, float3 rcPrevWo, float3 pathThroughput)
    
    {
        this.rcPrevHit = rcPrevHit;
        this.rcPrevWo = rcPrevWo;
        this.pathThroughput = pathThroughput;
    }

    [mutating]
    void initEmpty()
    {
        rcPrevHit = {};        
        rcPrevWo = float3(0.f);
        pathThroughput = float3(0.f);
    }

    [mutating]
    void initPseudoData()
    {

        rcPrevHit.data = {};
        rcPrevHit.data[0] = uint(HitType::Triangle) << HitInfo::kTypeOffset;

        rcPrevWo = float3(0.f);
        pathThroughput = float3(1.f);
    }
}

struct NeighborValidMask
{
    uint data;
    bool isValid(int i)
    {
        return data >> i & 1;
    }

    [mutating]
    void setValid(int i, bool val)
    {
        data &= ~(1 << i);
        data |= int(val) << i;
    }
}
