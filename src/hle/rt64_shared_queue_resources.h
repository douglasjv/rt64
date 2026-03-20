//
// RT64
//

#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_emulator_configuration.h"
#include "common/rt64_enhancement_configuration.h"
#include "render/rt64_render_target_manager.h"

#include "rt64_framebuffer_manager.h"

#if RT_ENABLED
#   include "render/rt64_raytracing_resources.h"
#endif

namespace RT64 {
    struct InterpolatedFrameCounters {
        uint32_t presented = 0;
        uint32_t available = 0;
        uint32_t count = 0;
        bool skipped = false;
    };

    struct AndroidVISnapshot {
        uint32_t address = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint8_t siz = 0;
        uint64_t sequence = 0;
        std::vector<uint8_t> bytes;
    };

    struct SharedQueueResources {
        // Render configuration.
        uint32_t swapChainWidth = 0;
        uint32_t swapChainHeight = 0;
        uint32_t swapChainRate = 0;
        hlslpp::float2 resolutionScale = 1.0f;
        uint32_t targetRate = 0;
        UserConfiguration userConfig;
        EnhancementConfiguration enhancementConfig;
        EmulatorConfiguration emulatorConfig;
        bool swapChainSizeChanged = false;
        bool rtConfigChanged = true;
        bool userConfigChanged = true;
        bool fbConfigChanged = true;
        bool newConfigValidated = false;
        std::mutex configurationMutex;

        // Render resource managers.
        FramebufferManager framebufferManager;
        RenderTargetManager renderTargetManager;
        std::mutex managerMutex;

        // Workload to present state.
        uint32_t viOriginalRate = 0;
        std::vector<uint32_t> colorImageAddressVector;
        std::unordered_set<uint32_t> colorImageAddressSet;
        std::vector<std::unique_ptr<RenderTarget>> interpolatedColorTargets;
        InterpolatedFrameCounters interpolatedFrames[2];
        uint32_t interpolatedFramesIndex = 0;
        std::mutex interpolatedMutex;
        std::condition_variable interpolatedCondition;
        std::mutex workloadMutex;
        std::mutex presentMutex;
        std::condition_variable presentCondition;
        uint32_t activePresents = 0;
        bool managerReconfiguration = false;
        AndroidVISnapshot androidVISnapshot;
        std::mutex androidVISnapshotMutex;

        void beginPresentResourceUse() {
            std::unique_lock<std::mutex> presentLock(presentMutex);
            presentCondition.wait(presentLock, [this]() {
                return !managerReconfiguration;
            });
            activePresents++;
        }

        void endPresentResourceUse() {
            {
                std::scoped_lock<std::mutex> presentLock(presentMutex);
                assert(activePresents > 0);
                activePresents--;
            }

            presentCondition.notify_all();
        }

        void beginManagerReconfiguration() {
            std::unique_lock<std::mutex> presentLock(presentMutex);
            managerReconfiguration = true;
            presentCondition.wait(presentLock, [this]() {
                return activePresents == 0;
            });
        }

        void endManagerReconfiguration() {
            {
                std::scoped_lock<std::mutex> presentLock(presentMutex);
                managerReconfiguration = false;
            }

            presentCondition.notify_all();
        }

        void setSwapChainSize(uint32_t width, uint32_t height) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            if ((swapChainWidth != width) || (swapChainHeight != height)) {
                swapChainWidth = width;
                swapChainHeight = height;
                swapChainSizeChanged = true;
            }
        }

        void setSwapChainRate(uint32_t rate) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            swapChainRate = rate;
        }

        void setUserConfig(const UserConfiguration &newUserConfig, bool discardFBs) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            userConfig = newUserConfig;
            userConfigChanged = true;
            fbConfigChanged = discardFBs;
            newConfigValidated = false;
        }

        void setEmulatorConfig(const EmulatorConfiguration &newEmulatorConfig) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            emulatorConfig = newEmulatorConfig;
        }

        void setEnhancementConfig(const EnhancementConfiguration &newEnhancementConfig) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            enhancementConfig = newEnhancementConfig;
        }

        void updateMultisampling(const RenderMultisampling &multisampling) {
            interpolatedFrames[0].skipped = true;
            interpolatedFrames[1].skipped = true;
            renderTargetManager.destroyAll();
            renderTargetManager.setMultisampling(multisampling);
        }

#   if RT_ENABLED
        RaytracingConfiguration rtConfig;

        void setRtConfig(const RaytracingConfiguration &newRtConfig) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            rtConfig = newRtConfig;
            rtConfigChanged = true;
        }
#   endif
    };
};
