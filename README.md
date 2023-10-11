# Conditional ReSTIR Prototype
![](teaser.png)

## Introduction
- This repo includes the source code for the following SIGGRAPH Asia 2023 paper

> **Conditional Resampled Importance Sampling and ReSTIR**<br>
> Markus Kettunen* (NVIDIA), Daqi Lin* (NVIDIA), Ravi Ramamoorthi (NVIDIA and UC San Diego), Thomas Bashford-Rogers (University of Warwick),  Chris Wyman (NVIDIA)<br>
> (*Joint first authors) <br>

Conditional ReSTIR defers ReSTIR-based path reuse by one or more bounces. It is based on Conditional Resampling Importance Sampling (CRIS) theory, an extension of GRIS [Lin et al. 2022] to conditional path spaces that enables reusing subpaths from unidirectional-sampled paths with correct unbiased contribution weights. Our conditional ReSTIR prototype modifies ReSTIR PT [Lin et al. 2022] with a final gather pass. As in photon mapping, such a final gather reduces blotchy artifacts from sample correlation.

- The method is implemented as a rendering component called "ConditionalReSTIR" (`Source/Falcor/Rendering/ConditionalReSTIR`) in Falcor 5.2.
See README_Falcor.md for the original README file provided by Falcor.
- A script `runConditionalReSTIRDemo.bat` is provided to show how the method works VeachAjar scene (from [Benedikt Bitterli's rendering resources](https://benedikt-bitterli.me/resources/)) which is contained in the repo.
- Before running the scripts, you need to compile the program and download the scene files following the instruction below.

## Prerequisites
- Windows 10 version 20H2 or newer
- Visual Studio 2022
- [Windows 10 SDK version 10.0.19041.1 Or Newer] (https://developer.microsoft.com/en-us/windows/downloads/sdk-archive)
- NVIDIA driver 466.11 or newer
- A GPU which supports DirectX Raytracing (recommended: NVIDIA GeForce RTX 4090)

## How to compile
- IMPORTANT: you have to git clone the repo before compiling!!! Otherwise, the script won't work.
- Click setup_vs2022.bat
- Open `build/windows-vs2022/Falcor.sln` and Build Solution in Release configuration 

## Run the demo
- execute `runConditionalReSTIRDemo.bat`
- The GUI contains self-explanatory settings for parameter tweaking.  

## Test with more scenes
- You can test your custom scene by running `build/windows-vs2022/bin/Release/Mogwai.exe` first, then load a scene file.
- A Falcor pyscene is recommended. For how to create a pyscene, please check out the `data/VeachAjar/VeachAjar.pyscene` as a template.
Details can be found in Falcor's [documentation](Docs/Usage/Scene-Formats.md)
- Alternatively, if you have a scene file with well defined lighting, material, and camera information that is supported by Falocr (like FBX), you can also create a one-line pyscene file, e.g. `sceneBuilder.importScene(YOUR_SCENE_FILE)`.

