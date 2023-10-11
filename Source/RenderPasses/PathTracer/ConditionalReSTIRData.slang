/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#include "Utils/Math/MathConstants.slangh"
__exported import StaticParams;
__exported import Rendering.ConditionalReSTIR.PathReservoir;
__exported import Rendering.ConditionalReSTIR.ConditionalReSTIR;

// maximum: 160 bytes
struct ConditionalReSTIRData
{
    PathRisState pathRis;
    PathRcInfo pathRcInfo;         
    float lastScatterPdf;
    float3 lastScatterWeight;
    float rrPdf;
    float pathTotalLength;

    __init()
    {
        pathRis = PathRisState();
        pathRcInfo = {};
        lastScatterPdf = 0.f;
        lastScatterWeight = 0.f;
        rrPdf = 1.f;
        pathTotalLength = 0.f;
    }
};
