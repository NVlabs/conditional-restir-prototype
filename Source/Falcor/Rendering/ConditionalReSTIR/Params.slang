/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

#ifndef HOST_CODE

/** Per-sample color format.
 */
enum class ColorFormat : uint32_t
{
    RGBA32F = 0,
    LogLuvHDR = 1,
};

enum class MISHeuristic : uint32_t
{
    Balance = 0,  ///< Balance heuristic.
    PowerTwo = 1, ///< Power heuristic (exponent = 2.0).
    PowerExp = 2, ///< Power heuristic (variable exponent).
};

// Define tile sizes in pixels.
// The frame is divided into tiles stored in scanline order, with pixels in tiles enumerated in Morton order.
static const uint2 kScreenTileDim = { 16, 16 }; ///< Screen-tile dimension in pixels.
static const uint2 kScreenTileBits = { 4, 4 };  ///< Bits needed to describe pixel position within a screen-tile.

// Define path configuration limits.
static const uint kMaxSamplesPerPixel = 128;      ///< Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxFrameDimension = 4096;     ///< Maximum supported frame dimension in pixels along x or y. We can increase the bit allocation if needed.
static const uint kMaxBounces = 254;             ///< Maximum supported number of bounces per bounce category (value 255 is reserved for internal use). The resulting path length may be longer than this.
static const uint kMaxLightSamplesPerVertex = 8; ///< Maximum number of shadow rays per path vertex for next-event estimation.

// Import static specialization constants.
__exported import StaticParams;
__exported import Utils.Math.BitTricks;
#endif

/** Path tracer parameters shared between host/device code.
 */
struct ReSTIRPathTracerParams
{
    int useFixedSeed = false; ///< Use fixed random seed. This is useful for debugging.
    uint fixedSeed = 1;       ///< The seed to use when 'useFixedSeed' is enabled.
    float lodBias = 0.f;      ///< LOD bias applied to secondary hits.

    float specularRoughnessThreshold = 0.2f; ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.

    // Runtime values
    uint2 frameDim = { 0, 0 };    ///< Frame dimension in pixels.
    uint2 screenTiles = { 0, 0 }; ///< Number of screen-tiles. Screen tiles may extend outside the frame.

    uint frameCount = 0; ///< Frames rendered. This is used as random seed.
    uint seed = 0;       ///< Random seed. This will get updated from the host depending on settings.
    int samplesPerPixel = 1;
    int DIMode = 0;

    int    useConditionalReSTIR = true;
    int3    pad;

#ifndef HOST_CODE
    bool disableDirectIllumination() { return DIMode >= 1; }
    bool disableGeneralizedDirectIllumination() { return DIMode == 2; }

    /** Computes the offset into the tiled sample buffer for a given tile.
        The samples for all pixels are stored consecutively after this offset.
        \param[in] tile Tile coordinates.
        \return Offset into tiled sample buffer.
    */
    uint getTileOffset(const uint2 tile)
    {
        uint maxSpp = samplesPerPixel > 0 ? samplesPerPixel : kMaxSamplesPerPixel;
        uint stride = kScreenTileDim.x * kScreenTileDim.y * maxSpp;
        uint tileIdx = tile.y * screenTiles.x + tile.x;
        return tileIdx * stride;
    }

    /** Computes the offset into the tiled sample buffer for a given pixel.
        \param[in] pixel Pixel on screen.
        \param[in] sampleOffset Per-pixel sample offset within tiles. Only used if samplesPerPixel == 0.
        \return Offset into tiled sample buffer.
    */
    uint getSampleOffset(const uint2 pixel, Texture2D<uint> sampleOffset)
    {
        uint2 tileID = pixel >> kScreenTileBits;
        uint tileOffset = getTileOffset(tileID);

        if (samplesPerPixel > 0)
        {
            uint tileBits = kScreenTileBits.x + kScreenTileBits.y;
            uint pixelIdx = interleave_16bit(pixel) & ((1 << tileBits) - 1); // TODO: Use interleave_8bit() if kScreenTileBits <= 4.
            return tileOffset + pixelIdx * samplesPerPixel;
        }
        else
        {
            return tileOffset + sampleOffset[pixel];
        }
    }

    // assume samplesPerPixel == 1
    uint getReservoirOffset(const uint2 pixel)
    {
        uint2 tileID = pixel >> kScreenTileBits;
        uint stride = kScreenTileDim.x * kScreenTileDim.y;
        uint tileIdx = tileID.y * screenTiles.x + tileID.x;
        uint tileOffset = tileIdx * stride;
        uint tileBits = kScreenTileBits.x + kScreenTileBits.y;
        uint pixelIdx = interleave_16bit(pixel) & ((1 << tileBits) - 1); // TODO: Use interleave_8bit() if kScreenTileBits <= 4.
        uint reservoirIdx = tileOffset + pixelIdx;
        return reservoirIdx;
    }

#endif
};

END_NAMESPACE_FALCOR
