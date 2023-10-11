/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
__exported import Utils.Sampling.SampleGeneratorInterface;

/** this sample generator stores 3 precomputed random numbers
*/
public struct TemporarySampleGenerator : ISampleGenerator
{
    struct Padded
    {
        TemporarySampleGenerator internal;
    }

    float mutate(float value, float sampleForMutation)
    {
        // return value;
        //  from pssmlt_sampler in Mitsuba
        float mutationSizeLow = 1.f / 65536.f;
        float mutationSizeHigh = 1.f / 1024.f;
        float logRatio = -log(mutationSizeHigh / mutationSizeLow);

        float sample = sampleForMutation;
        bool add;

        if (sample < 0.5f) {
            add = true;
            sample *= 2.0f;
        }
        else {
            add = false;
            sample = 2.0f * (sample - 0.5f);
        }

        float dv = mutationSizeHigh * exp(sample * logRatio);
        if (add) {
            value += dv;
            if (value > 1)
                value -= 1;
        }
        else {
            value -= dv;
            if (value < 0)
                value += 1;
        }
        return value;
    }

    [mutating] void mutate(float3 samplesForMutation)
    {
        rands.x = mutate(rands.x, samplesForMutation.x);
        rands.y = mutate(rands.y, samplesForMutation.y);
        rands.z = mutate(rands.z, samplesForMutation.z);
    }

    __init(float3 _rands)
    {
        this.rands = _rands;
        idx = 0;
    }

    /** Returns the next sample value. This function updates the state.
    */
    [mutating] uint next()
    {
        return 0;
    }

    [mutating] float nextFloat()
    {
        return rands[idx++];
    }

    [mutating] float sampleNext1D()
    {
        return nextFloat();
    }

    float3 rands;
    int idx;
};
