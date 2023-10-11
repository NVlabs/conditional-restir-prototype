/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
import Utils.Attributes;

struct RetraceWorkloadQueue
{
    [root] RWByteAddressBuffer counter;
    RWByteAddressBuffer workload;
    RWByteAddressBuffer workloadExtra;

    uint addCounter(uint value)
    {
        uint sum = WaveActiveSum(value);
        uint offset = WavePrefixSum(value);
        uint originalValue;
        if (WaveIsFirstLane())
        {
            counter.InterlockedAdd(0, sum, originalValue);
        }
        originalValue = WaveReadLaneFirst(originalValue); 
        return originalValue + offset;
    }
}
