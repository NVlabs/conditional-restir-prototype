/***************************************************************************
# Copyright (c) 2023, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, see LICENSE.md
**************************************************************************/
#pragma once
#include <string>
#include <fstream>
#include <vector>
#include "Falcor.h"

using namespace Falcor;

class UserInteractionRecorder
{
public:
    UserInteractionRecorder() {};

    void recordStep(Scene::SharedPtr pScene)
    {
        if (mRecordUserInteraction)
        {
            if (!mUserInteractionOutputFile.is_open())
            {
                std::time_t result = std::time(nullptr);
                char filename_char[100];
                std::strftime(filename_char, 100, "%Y-%m-%d-%H-%M-%S", std::localtime(&result));
                std::string filename(filename_char);
                mUserInteractionFileName = filename;
                filename += "_CamCapture.txt";
                mUserInteractionOutputFile.open(mOutputFolderName + "/" + filename);
                mRecordedFrameCount = 0;
            }

            float3 pos = pScene->getCamera()->getPosition();
            float3 target = pScene->getCamera()->getTarget();
            float3 up = pScene->getCamera()->getUpVector();

            mRecordedFrameCount++;
            mUserInteractionOutputFile << pos.x << " " << pos.y << " " << pos.z << std::endl;
            mUserInteractionOutputFile << target.x << " " << target.y << " " << target.z << std::endl;
            mUserInteractionOutputFile << up.x << " " << up.y << " " << up.z << std::endl << std::endl;

            if (mNumWarmupFramesReplicated > 0 && mRecordedFrameCount == 1)
            {
                for (int i = 0; i < mNumWarmupFramesReplicated; i++)
                {
                    mRecordedFrameCount++;
                    mUserInteractionOutputFile << pos.x << " " << pos.y << " " << pos.z << std::endl;
                    mUserInteractionOutputFile << target.x << " " << target.y << " " << target.z << std::endl;
                    mUserInteractionOutputFile << up.x << " " << up.y << " " << up.z << std::endl << std::endl;
                }
            }
        }
    }

    bool replayStep(Scene::SharedPtr pScene)
    {
        bool shouldFreeze = false;
        if (mReplayUserInteraction && mRecordedFrameCount < mCameraTargetSequence.size())
        {
            pScene->getCamera()->setTarget(mCameraTargetSequence[mRecordedFrameCount]);
            pScene->getCamera()->setPosition(mCameraPositionSequence[mRecordedFrameCount]);

            shouldFreeze = mRecordedFrameCount == mFrozenFrameId;
        }
        else if (mReplayUserInteraction)
        {
            mReplayUserInteraction = false;
            mRecordedFrameCount = -1;
        }

        mRecordedFrameCount++;
        return shouldFreeze;
    }

    // return if need to flag OptionsChanged
    bool renderUI(Gui::Widgets& widget)
    {
        bool needToFlag = false;

        widget.text("Output Directory\n" + mOutputFolderName);
        if (widget.button("Change Folder"))
        {
            std::filesystem::path path;
            if (chooseFolderDialog(path))
            {
                std::filesystem::path path_ = path;
                if (path_.is_absolute())
                {
                    // Use relative path to executable directory if possible.
                    auto relativePath = path_.lexically_relative(getRuntimeDirectory());
                    if (!relativePath.empty() && relativePath.string().find("..") == std::string::npos) path_ = relativePath;
                }
                mOutputFolderName = path_.string();
            }
        }

        if (widget.button(mRecordUserInteraction ? "Stop Record" : "Record User Interaction"))
        {
            mRecordUserInteraction = !mRecordUserInteraction;
            if (mRecordUserInteraction) needToFlag = true;
            if (!mRecordUserInteraction)
            {
                mRecordedFrameCount = 0;
                if (mUserInteractionOutputFile.is_open())
                    mUserInteractionOutputFile.close();
            }
        }

        widget.var("Num Warmup Frames To Replicate", mNumWarmupFramesReplicated, 0, 100);

        if (widget.button("Load User Input File"))
        {
            mCameraTargetSequence.clear();
            mCameraPositionSequence.clear();
            std::filesystem::path filename;
            FileDialogFilterVec txtFile;
            txtFile.push_back({ "txt", "txt file" });
            if (openFileDialog(txtFile, filename))
            {
                if (filename.string().find("CamCapture_") != std::string::npos)
                {
                    std::string temp = filename.string().substr(filename.string().find("CamCapture_") + 11);
                    std::string val = temp.substr(0, temp.find("."));
                    //std::cout << std::stoi(val) << std::endl;
                    mFrozenFrameId = std::stoi(val);
                }

                std::ifstream f(filename);
                // load camera positions and targets
                while (!f.eof())
                {
                    float x, y, z;
                    f >> x >> y >> z;
                    if (f.fail() || f.eof()) break;
                    mCameraPositionSequence.push_back(float3(x, y, z));
                    f >> x >> y >> z;
                    mCameraTargetSequence.push_back(float3(x, y, z));
                    f >> x >> y >> z;
                    mCameraUpSequence.push_back(float3(x, y, z));
                }
                std::cout << "Loaded " << mCameraPositionSequence.size() << " frames user input" << std::endl;

                // replicate first 20 frames (such that ReSTIR starts at a warmed up state)
                //for (int i = 0; i < 20; i++)
                //{
                //    mCameraPositionSequence.insert(mCameraPositionSequence.begin(), mCameraPositionSequence[0]);
                //    mCameraTargetSequence.insert(mCameraTargetSequence.begin(), mCameraTargetSequence[0]);
                //    mCameraUpSequence.insert(mCameraUpSequence.begin(), mCameraUpSequence[0]);
                //}

                f.close();
            }
        }

        if (widget.button(mReplayUserInteraction ? "Stop Playing User Input" : "Play Loaded User Input"))
        {
            if (mReplayUserInteraction) needToFlag = true;
            mReplayUserInteraction = !mReplayUserInteraction;
            mRecordedFrameCount = 0;
        }

        widget.var("Freeze Frame ID", mFrozenFrameId, -1, (int)mCameraPositionSequence.size(), 1);

        return needToFlag;
    }

    // user interaction
    bool                            mRecordUserInteraction = false;
    int                             mRecordedFrameCount = 0;
    int                             mFrozenFrameId = -1;
    std::string                     mUserInteractionFileName = "";
    std::ofstream                   mUserInteractionOutputFile;
    int                             mNumWarmupFramesReplicated = 10;

    // replay user interaction
    std::vector<float3> mCameraPositionSequence;
    std::vector<float3> mCameraTargetSequence;
    std::vector<float3> mCameraUpSequence;
    bool mReplayUserInteraction = false;
    std::string mOutputFolderName = ".";
    std::string mBaseCaptureFileName = "output";
};
