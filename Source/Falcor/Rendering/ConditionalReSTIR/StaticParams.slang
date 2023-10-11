/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
__exported import Rendering.Materials.TexLODTypes;

/** Translation of defines set by the host to compile-time constants used to
    configure the path tracer without overhead from dynamic control flow.
    This will eventually be replaced by specialization constants in Slang.
*/

// #ifndef SAMPLES_PER_PIXEL
// #error StaticParams.slang is missing defines
// #endif

static const uint kUseViewDir = USE_VIEW_DIR;
static const uint kMaxSurfaceBounces = MAX_SURFACE_BOUNCES;
static const uint kMaxDiffuseBounces = MAX_DIFFUSE_BOUNCES;
static const uint kMaxSpecularBounces = MAX_SPECULAR_BOUNCES;
static const uint kMaxTransmissionBounces = MAX_TRANSMISSON_BOUNCES;
static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
static const bool kUseBSDFSampling = USE_BSDF_SAMPLING;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;
static const bool kUseRussianRoulette = USE_RUSSIAN_ROULETTE;
static const bool kUseRTXDI = USE_RTXDI;
static const bool kUseAlphaTest = USE_ALPHA_TEST;
static const bool kUseLightsInDielectricVolumes = USE_LIGHTS_IN_DIELECTRIC_VOLUMES;
static const bool kDisableCaustics = DISABLE_CAUSTICS;
static const TexLODMode kPrimaryLodMode = TexLODMode(PRIMARY_LOD_MODE);
static const bool kUseNRDDemodulation = USE_NRD_DEMODULATION;
static const uint kColorFormat = COLOR_FORMAT;
static const uint kMISHeuristic = MIS_HEURISTIC;
static const float kMISPowerExponent = MIS_POWER_EXPONENT;

