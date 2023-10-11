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
__exported import Rendering.Materials.LobeType;
__exported import Utils.Sampling.SampleGeneratorInterface;
__exported import Rendering.Materials.BxDF; // For StandardBSDFData
#include "Utils/HostDeviceShared.slangh" // For bit packing macros

__exported import Scene.ShadingData;
__exported import Rendering.Materials.IBSDF; // For kMinCosTheta
__exported import Rendering.Materials.LobeType;
__exported import Utils.Sampling.SampleGeneratorInterface;

/** Describes a BSDF sample.
*/
struct BSDFSample
{
    float3  wo;             ///< Sampled direction in world space (normalized).
    float   pdf;            ///< pdf with respect to solid angle for the sampled direction (wo).
    float3  weight;         ///< Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
    uint    lobeType;       ///< Sampled lobeType. This is a combination of LobeType flags (see LobeType.slang).

    bool isLobe(LobeType type)
    {
        return (lobeType & uint(type)) != 0;
    }

    [mutating] uint getComponentType()
    {
        uint componentType = 0;
        if (isLobe(LobeType::DiffuseReflection) && (isLobe(LobeType::Specular) || isLobe(LobeType::Delta)))
        { // we use this to differentiate SpecularReflectionTransmission from SpecularReflection
            // for Standard Material Only
            componentType = 3;
            lobeType &= ~((uint)LobeType::DiffuseReflection);
        }
        else if (!isLobe(LobeType::DiffuseReflection))
        {
            componentType = isLobe(LobeType::DiffuseTransmission)
                                ? 1 : componentType;
            componentType = (isLobe(LobeType::Specular) || isLobe(LobeType::Delta))
                                ? 2 : componentType;
        }
        return componentType;
    }
};

/** Describes BSDF properties.

    These properties may be useful as input to denoisers and other perceptual techniques.
    The BSDF implementations are expected to provide a best fit approximation.
    The values are an average of the respective BSDFs if a material instance holds multiple.
*/
struct BSDFProperties
{
    float3  emission;                       ///< Radiance emitted in the incident direction (wi).
    float   roughness;                      ///< Surface roughness on a perceptually linear scale, where 0.0 = perfectly smooth and 1.0 = maximum roughness.

    // Approximate directional-hemispherical reflectance/transmittance of the BSDF (black-sky albedo).
    // The exact values are given by the integrals of the BSDF over wo given an incident direction wi.
    // The terms are separated into diffuse and non-diffuse components. Due to energy conservation, the sum is expected to be <= 1.0.
    float3  diffuseReflectionAlbedo;        ///< Directional-hemispherical diffuse reflectance. This is the ratio of total energy diffusely reflected to the energy incident along wi.
    float3  diffuseTransmissionAlbedo;      ///< Directional-hemispherical diffuse transmittance. This is the ratio of total energy diffusely transmitted to the energy incident along wi.
    float3  specularReflectionAlbedo;       ///< Directional-hemispherical specular reflectance. This is the ratio of total energy non-diffusely reflected to the energy incident along wi.
    float3  specularTransmissionAlbedo;     ///< Directional-hemispherical specular transmittance. This is the ratio of total energy non-diffusely transmitted to the energy incident along wi.

    // Approximate specular reflectance. This is the color of specular reflection.
    // The diffuse reflectance is approximated as a Lambertian given by the diffuse albedo above.
    float3  specularReflectance;            ///< Specular reflectance at normal incidence (F0). This is in the range [0,1].

    uint    flags;                          ///< Flags storing additional properties.

    enum Flags : uint
    {
        IsTransmissive = 0x1,
    };

    property isTransmissive : bool
    {
        get { return (flags & (uint)Flags::IsTransmissive) != 0; }
    }
};

#ifndef FALCOR_MATERIAL_INSTANCE_SIZE
#define FALCOR_MATERIAL_INSTANCE_SIZE 76
#endif

/** Interface for a bidirection scattering distribution function (BSDF).

    The term BSDF is used here in a broad sense for the mathematical function which
    describes the scattering of light at a shading location.

    This interface allows evaluation and sampling of the BSDF, and associated functionality.
    Implementations of the interface has all material properties evaluated at the shading location.
    BSDF instances are created and initialized by the corresponding material.

    The shading location and its attributes are described by a ShadingData struct.
    The ShadingData view direction field specifies the direction from which the
    shading location is seen. By convention we call this the incident direction (wi).
    The outgoing direction (wo) is the direction in which the transported quantity
    (radiance or importance) is scattered in.

    Conventions:
     - The incident and outgoing direction point away from the shading location.
     - The incident direction (wi) is given by ShadingData.
     - The outgoing direction (wo) is sampled.
     - The foreshortening term with respect to the sampled direction is always included.

     All calculations are done in world coordinate frame. MaterialInstances will convert
     data into local coordinate frame before passing it to the respective BSDFs.

    Note:
     - The [anyValueSize(n)] attribute specifies the maximum size in bytes an implementation type
       to IMaterialInstance can be. Slang compiler will check the implementation types and emits an error
       if an implementation exceeds this size.
     - The maximum size can be increased if needed, but it should be kept as small as possible
       to reduce register pressure in case the compiler fails to optimize.
*/
[anyValueSize(FALCOR_MATERIAL_INSTANCE_SIZE)] // TODO: Reduce to 64B
interface IMaterialInstance
{
    /** Evaluates the material instance.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \param[in,out] sg Sample generator.
        \return Returns f(wi, wo) * dot(wo, n).
    */
    // Minimum cos(theta) for the incident and outgoing vectors.
    // Some BSDF functions are not robust for cos(theta) == 0.0,
    // so using a small epsilon for consistency.
    // TODO: Move into IBSDF if possible
    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg);

    // get lobe pmf
    float getLobePmf(const ShadingData sd, uint lobeMask);

    /** Samples the material instance.
        \param[in] sd Shading data.
        \param[in,out] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \param[in] useImportanceSampling Hint to use importance sampling, else default to reference implementation if available.
        \return True if a sample was generated, false otherwise.
    */
    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true);

    /** Evaluates the directional pdf for sampling the given direction.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \param[in] useImportanceSampling Hint to use importance sampling, else default to reference implementation if available.
        \return PDF (of the masked lobes) with respect to solid angle for sampling direction wo (0 for delta events).
    */
    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true);

    /** Evaluates the directional pdf for sampling the given direction (assuming importance sampling).
        \param[in]  sd Shading data.
        \param[in]  wo Outgoing direction.
        \param[out] pdfAll pdf of all components.
        \return PDF (of the masked lobes) with respect to solid angle for sampling direction wo (0 for delta events).
    */
    float evalPdf(const ShadingData sd, const float3 wo, out float pdfAll);

    /** Evaluates the sum of bsdfs and pdfs of lobes specified by the lobeMask. (assuming useImportanceSampling == true)
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \param[out] pdf sum of pdfs of selected lobes.
        \param[out] pdfAllLobes sum of pdfs of all lobes (directional pdf for sampling the given direction).
        \return Returns sum of f(wi, wo) * dot(wo, n) of selected lobes.
    */
    float3 evalBsdfAndPdf<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg, uint lobeMask, out float pdf, out float pdfAllLobes);

    float3 getDiffuse();

    float3 getSpecular();

    float4 getSpecRoughMetalEta();

    /** Return BSDF properties.
        \param[in] sd Shading data.
        \return A struct with properties.
    */
    BSDFProperties getProperties(const ShadingData sd);

    /** Return the set of available BSDF lobes.
        \param[in] sd Shading data.
        \return A combination of LobeType flags (see LobeType.slang).
    */
    uint getLobeTypes(const ShadingData sd);
}
