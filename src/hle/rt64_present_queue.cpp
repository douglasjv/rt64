//
// RT64
//

#include "rt64_present_queue.h"

#include <cstring>

#include "common/rt64_thread.h"
#include "rhi/rt64_render_hooks.h"

#include "rt64_workload_queue.h"

#if defined(__ANDROID__) && defined(BANJO_ENABLE_ANDROID_TRACE_LOGS)
#include <android/log.h>
#define BANJO_ANDROID_LOG(...) __android_log_print(ANDROID_LOG_INFO, "BanjoRecomp", __VA_ARGS__)
#else
#define BANJO_ANDROID_LOG(...) ((void)0)
#endif

namespace RT64 {
    // PresentQueue

#if defined(__ANDROID__)
    static uint32_t g_android_present_draw_logs = 0;
    static uint32_t g_android_direct_vi_upload_logs = 0;
    static uint32_t g_android_interesting_direct_vi_upload_logs = 0;
    static uint32_t g_android_gameplay_direct_vi_metadata_logs = 0;
    static uint32_t g_android_gameplay_origin_candidate_logs = 0;
    static uint32_t g_android_direct_vi_upload_last_address = UINT32_MAX;
    static uint32_t g_android_direct_vi_upload_last_width = 0;
    static uint32_t g_android_direct_vi_upload_last_height = 0;
    static uint32_t g_android_direct_vi_upload_last_offset = UINT32_MAX;
    static uint8_t g_android_direct_vi_upload_last_siz = UINT8_MAX;
    static uint32_t g_android_gameplay_direct_vi_last_address = UINT32_MAX;
    static uint64_t g_android_gameplay_direct_vi_last_timestamp = 0;
#endif

    PresentQueue::PresentQueue() {
        reset();
    }

    PresentQueue::~PresentQueue() {
        presentThreadRunning = false;
        cursorCondition.notify_all();

        if (presentThread != nullptr) {
            presentThread->join();
            delete presentThread;
        }

        presentIdCondition.notify_all();
    }

    void PresentQueue::reset() {
        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = 0;
        presentId = 0;
    }

    void PresentQueue::advanceToNextPresent() {
        int nextWriteCursor = (writeCursor + 1) % presents.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a present being used by the GPU.
        bool waitForBarrier;
        do {
            const std::scoped_lock lock(cursorMutex);
            waitForBarrier = (nextWriteCursor == barrierCursor);
        } while (waitForBarrier);

        // Modify the cursor and notify anything waiting on the queue.
        {
            const std::scoped_lock lock(cursorMutex);
            writeCursor = nextWriteCursor;
        }

        cursorCondition.notify_all();
    }

    void PresentQueue::repeatLastPresent() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t PresentQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(presents.size()) - 1;
        }
    }

    void PresentQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void PresentQueue::waitForPresentId(uint64_t waitId) {
        std::unique_lock<std::mutex> presentLock(presentIdMutex);
        presentIdCondition.wait(presentLock, [&]() {
            return (waitId <= presentId) || !presentThreadRunning;
        });
    }

    void PresentQueue::setup(const External &ext) {
        this->ext = ext;

        viRenderer = std::make_unique<VIRenderer>();

        presentThreadRunning = true;
        presentThread = new std::thread(&PresentQueue::threadLoop, this);
    }

    void PresentQueue::threadPresent(const Present &present, bool &swapChainValid) {
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        const bool usingMSAA = (targetManager.multisampling.sampleCount > 1);
        hlslpp::float2 resolutionScale;
        EnhancementConfiguration::Presentation::Mode presentationMode;
        bool removeBlackBorders;
        UserConfiguration::RefreshRate refreshRate;
        UserConfiguration::Filtering filtering;
        uint32_t viOriginalRate;
        uint32_t targetRate;
        {
            std::scoped_lock<std::mutex> configurationLock(ext.sharedResources->configurationMutex);
            resolutionScale = ext.sharedResources->resolutionScale;
            presentationMode = ext.sharedResources->enhancementConfig.presentation.mode;
            removeBlackBorders = ext.sharedResources->enhancementConfig.presentation.removeBlackBorders;
            refreshRate = ext.sharedResources->userConfig.refreshRate;
            filtering = ext.sharedResources->userConfig.filtering;
            viOriginalRate = ext.sharedResources->viOriginalRate;
            targetRate = ext.sharedResources->targetRate;
        }

        RenderTarget *colorTarget = nullptr;
        int32_t framesToPresent = 1;
        hlslpp::float2 renderTextureCoordinateOffset = { 0.0f, 0.0f };
        bool lockedWorkloadMutex = false;
        InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
        struct PresentResourceGuard {
            SharedQueueResources *sharedResources = nullptr;

            explicit PresentResourceGuard(SharedQueueResources *sharedResources) : sharedResources(sharedResources) {
                sharedResources->beginPresentResourceUse();
            }

            ~PresentResourceGuard() {
                sharedResources->endPresentResourceUse();
            }
        } presentResourceGuard(ext.sharedResources);

        // TODO: There's a possible race condition interactions that can happen while the workload
        // queue is rendering extra frames and the present event is processed while it's generating
        // interpolated frames. When the framebuffer manager or the render target manager maps are
        // modified while the present queue is retrieving the framebuffer or the target. These can
        // likely be solved by locking the access to the managers during modification.
        
        // Perform any external write operations indicated by the event.
        if (!present.fbOperations.empty()) {
            const std::scoped_lock lock(screenFbChangePoolMutex);
            {
                RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                fbManager.performOperations(ext.presentGraphicsWorker, &screenFbChangePool, nullptr, ext.shaderLibrary, nullptr,
                    present.fbOperations, targetManager, resolutionScale, 0, 0, nullptr);
            }
        }

        // Present the VI specified by the event.
        // Attempt to find the matching framebuffer for the VI based on the origin address.
        // If that fails, we look at the shared storage.
        if (present.screenVI.visible()) {
            auto lockWorkloadMutex = [&]() {
                if (!lockedWorkloadMutex) {
                    lockedWorkloadMutex = true;
                    ext.sharedResources->workloadMutex.lock();
                }
            };

            auto uploadPresentStorage = [&](RenderTarget &target, const uint8_t *storageBytes, size_t storageBytesCount, uint32_t fbAddress, const hlslpp::uint2 &fbSize, uint8_t fbSiz, uint8_t fbFmt) -> bool {
                if ((storageBytes == nullptr) || (storageBytesCount == 0) || (fbSize.x == 0) || (fbSize.y == 0) || (fbSiz < G_IM_SIZ_16b)) {
                    return false;
                }

                const size_t bytesPerPixel = size_t(1U << (fbSiz - 1));
                const size_t expectedBytes = size_t(fbSize.x) * size_t(fbSize.y) * bytesPerPixel;
                if (storageBytesCount < expectedBytes) {
                    return false;
                }

                lockWorkloadMutex();

                scratchFb.addressStart = fbAddress;
                scratchFb.width = fbSize.x;
                scratchFb.height = fbSize.y;
                scratchFb.siz = fbSiz;

                target.resize(ext.presentGraphicsWorker, scratchFb.width, scratchFb.height);
                target.resolutionScale = { 1.0f, 1.0f };
                target.downsampleMultiplier = 1;

                scratchFb.nativeTarget.resetBufferHistory();

                {
                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                    target.clearColorTarget(ext.presentGraphicsWorker);
                    FramebufferChange *colorFbChange = scratchFb.readChangeFromBytes(ext.presentGraphicsWorker, scratchFbChangePool, Framebuffer::Type::Color,
                        fbFmt, storageBytes, 0, scratchFb.height, ext.shaderLibrary);

                    if (colorFbChange != nullptr) {
                        target.copyFromChanges(ext.presentGraphicsWorker, *colorFbChange, scratchFb.width, scratchFb.height, 0, ext.shaderLibrary);
                    }
                }

                scratchFbChangePool.reset();
                return true;
            };

            auto uploadPresentStorageTarget = [&](const uint8_t *storageBytes, size_t storageBytesCount, uint32_t fbAddress, const hlslpp::uint2 &fbSize, uint8_t fbSiz, uint8_t fbFmt, const char *sourceTag) -> RenderTarget * {
                if ((storageBytes == nullptr) || (storageBytesCount == 0) || (fbSize.x == 0) || (fbSize.y == 0) || (fbSiz < G_IM_SIZ_16b)) {
                    return nullptr;
                }

                if (directStorageTarget == nullptr) {
                    directStorageTarget = std::make_unique<RenderTarget>(fbAddress, Framebuffer::Type::Color, RenderMultisampling(), false);
                }

                if ((directStorageTarget->width != fbSize.x) || (directStorageTarget->height != fbSize.y) || directStorageTarget->isEmpty()) {
                    directStorageTarget->setupColor(ext.presentGraphicsWorker, fbSize.x, fbSize.y);
                }

                directStorageTarget->resolutionScale = { 1.0f, 1.0f };
                directStorageTarget->downsampleMultiplier = 1;

                if (!uploadPresentStorage(*directStorageTarget, storageBytes, storageBytesCount, fbAddress, fbSize, fbSiz, fbFmt)) {
                    return nullptr;
                }

                return directStorageTarget.get();
            };

            auto uploadPresentStorageDirect = [&](const uint8_t *storageBytes, size_t storageBytesCount, uint32_t fbAddress, const hlslpp::uint2 &fbSize, uint8_t fbSiz, uint32_t storageOffset, const char *sourceTag, Framebuffer::Type sourceType, uint8_t sourceFmt) -> RenderTarget * {
#if defined(__ANDROID__)
                if ((storageBytes == nullptr) || (storageBytesCount == 0) || (fbSize.x == 0) || (fbSize.y == 0) || (fbSiz < G_IM_SIZ_16b)) {
                    return nullptr;
                }

                const size_t pixelCount = size_t(fbSize.x) * size_t(fbSize.y);
                const size_t bytesPerPixel = (fbSiz == G_IM_SIZ_16b) ? 2 : ((fbSiz == G_IM_SIZ_32b) ? 4 : 0);
                if (bytesPerPixel == 0) {
                    return nullptr;
                }

                const size_t expectedBytes = pixelCount * bytesPerPixel;
                if (storageBytesCount < expectedBytes) {
                    return nullptr;
                }

                if (directStorageTarget == nullptr) {
                    directStorageTarget = std::make_unique<RenderTarget>(fbAddress, Framebuffer::Type::Color, RenderMultisampling(), false);
                }

                if ((directStorageTarget->width != fbSize.x) || (directStorageTarget->height != fbSize.y) || directStorageTarget->isEmpty()) {
                    directStorageTarget->setupColor(ext.presentGraphicsWorker, fbSize.x, fbSize.y);
                }

                directStorageTarget->resolutionScale = { 1.0f, 1.0f };
                directStorageTarget->downsampleMultiplier = 1;

                thread_local std::vector<uint8_t> decodedRGBA8;
                decodedRGBA8.resize(pixelCount * 4);

                const uint8_t *src = storageBytes;
                if (fbSiz == G_IM_SIZ_16b) {
                    thread_local std::vector<uint8_t> wordSwappedRGBA16;
                    wordSwappedRGBA16.resize(expectedBytes);

                    const size_t wholeWords = expectedBytes / sizeof(uint32_t);
                    const uint32_t *srcWords = reinterpret_cast<const uint32_t *>(storageBytes);
                    uint32_t *dstWords = reinterpret_cast<uint32_t *>(wordSwappedRGBA16.data());
                    for (size_t i = 0; i < wholeWords; i++) {
#if defined(__GNUC__) || defined(__clang__)
                        dstWords[i] = __builtin_bswap32(srcWords[i]);
#else
                        dstWords[i] = _byteswap_ulong(srcWords[i]);
#endif
                    }

                    const size_t tailBytes = expectedBytes - (wholeWords * sizeof(uint32_t));
                    if (tailBytes > 0) {
                        std::memcpy(wordSwappedRGBA16.data() + wholeWords * sizeof(uint32_t), storageBytes + wholeWords * sizeof(uint32_t), tailBytes);
                    }

                    src = wordSwappedRGBA16.data();

                    for (size_t i = 0; i < pixelCount; i++) {
                        const size_t byteOffset = i * 2;
                        const uint16_t rgba16 = (uint16_t(src[byteOffset + 0]) << 8) | uint16_t(src[byteOffset + 1]);
                        const uint8_t r = uint8_t((rgba16 >> 11) & 0x1F);
                        const uint8_t g = uint8_t((rgba16 >> 6) & 0x1F);
                        const uint8_t b = uint8_t((rgba16 >> 1) & 0x1F);
                        decodedRGBA8[i * 4 + 0] = uint8_t((r << 3) | (r >> 2));
                        decodedRGBA8[i * 4 + 1] = uint8_t((g << 3) | (g >> 2));
                        decodedRGBA8[i * 4 + 2] = uint8_t((b << 3) | (b >> 2));
                        decodedRGBA8[i * 4 + 3] = (rgba16 & 1) ? 0xFF : 0x00;
                    }
                }
                else {
                    for (size_t i = 0; i < pixelCount; i++) {
                        const size_t byteOffset = i * 4;
                        decodedRGBA8[i * 4 + 0] = src[byteOffset + 0];
                        decodedRGBA8[i * 4 + 1] = src[byteOffset + 1];
                        decodedRGBA8[i * 4 + 2] = src[byteOffset + 2];
                        decodedRGBA8[i * 4 + 3] = src[byteOffset + 3];
                    }
                }

                {
                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                    directStorageTarget->uploadRGBA8(ext.presentGraphicsWorker, decodedRGBA8.data(), fbSize.x, fbSize.y);
                }

                const bool interestingDirectSource = (present.screenVI.width != 320U) || (uint32_t(fbSize.x) != 320U) || (uint32_t(fbSize.y) != 240U) ||
                    ((fbAddress != 0U) && (fbAddress < 0x00600000U)) || (storageOffset != 0U) || (std::strcmp(sourceTag, "vi") != 0);
                const bool uploadChanged = (fbAddress != g_android_direct_vi_upload_last_address) || (uint32_t(fbSize.x) != g_android_direct_vi_upload_last_width) ||
                    (uint32_t(fbSize.y) != g_android_direct_vi_upload_last_height) || (storageOffset != g_android_direct_vi_upload_last_offset) || (fbSiz != g_android_direct_vi_upload_last_siz);
                const bool shouldLogUpload = uploadChanged && ((g_android_direct_vi_upload_logs < 8U) ||
                    (interestingDirectSource && (g_android_interesting_direct_vi_upload_logs < 32U)));
                if (shouldLogUpload) {
                    BANJO_ANDROID_LOG("PresentQueue uploaded VI storage directly: source=%s address=0x%08X size=%ux%u siz=%u storageOffset=0x%X sourceType=%u sourceFmt=%u",
                        sourceTag, fbAddress, uint32_t(fbSize.x), uint32_t(fbSize.y), fbSiz, storageOffset, uint32_t(sourceType), uint32_t(sourceFmt));
                    if (interestingDirectSource) {
                        g_android_interesting_direct_vi_upload_logs++;
                    }
                    else {
                        g_android_direct_vi_upload_logs++;
                    }
                    g_android_direct_vi_upload_last_address = fbAddress;
                    g_android_direct_vi_upload_last_width = uint32_t(fbSize.x);
                    g_android_direct_vi_upload_last_height = uint32_t(fbSize.y);
                    g_android_direct_vi_upload_last_offset = storageOffset;
                    g_android_direct_vi_upload_last_siz = fbSiz;
                }
                else if (uploadChanged) {
                    g_android_direct_vi_upload_last_address = fbAddress;
                    g_android_direct_vi_upload_last_width = uint32_t(fbSize.x);
                    g_android_direct_vi_upload_last_height = uint32_t(fbSize.y);
                    g_android_direct_vi_upload_last_offset = storageOffset;
                    g_android_direct_vi_upload_last_siz = fbSiz;
                }

                return directStorageTarget.get();
#else
                return nullptr;
#endif
            };

            const uint32_t screenFbAddress = present.screenVI.fbAddress();
            const hlslpp::uint2 screenFbSize = present.screenVI.fbSize();
            const uint8_t screenFbSiz = present.screenVI.fbSiz();
            const uint32_t storageBaseAddress = present.storageAddress;
            const bool storageContainsScreen = !present.storage.empty() && (screenFbAddress >= storageBaseAddress) &&
                (size_t(screenFbAddress - storageBaseAddress) <= present.storage.size());
            const size_t screenStorageOffset = storageContainsScreen ? size_t(screenFbAddress - storageBaseAddress) : 0;
            const uint8_t *screenStorageBytes = storageContainsScreen ? (present.storage.data() + screenStorageOffset) : nullptr;
            const size_t screenStorageBytesCount = storageContainsScreen ? (present.storage.size() - screenStorageOffset) : 0;

            Framebuffer *directSourceFb = nullptr;
            uint32_t directSourceOffset = uint32_t(screenStorageOffset);
            hlslpp::uint2 directSourceSize = screenFbSize;
            uint32_t directSourceAddress = screenFbAddress;
            uint8_t directSourceSiz = screenFbSiz;
            const char *directSourceTag = "vi";
            Framebuffer::Type directSourceType = Framebuffer::Type::None;
            uint8_t directSourceFmt = UINT8_MAX;
            const uint8_t *directSourceBytes = screenStorageBytes;
            size_t directSourceBytesCount = screenStorageBytesCount;
            const uint8_t *directSourceBaseBytes = directSourceBytes;
            size_t directSourceBaseBytesCount = directSourceBytesCount;
            uint32_t directSourceBaseOffset = directSourceOffset;
            uint32_t directSourceBaseAddress = directSourceAddress;
            hlslpp::float2 directTextureCoordinateOffset = { 0.0f, 0.0f };
            Framebuffer *bestOriginFb = nullptr;
            Framebuffer *bestColorOriginFb = nullptr;

            if (!viewRDRAM && !present.storage.empty()) {
                auto isBetterOriginCandidate = [&](const Framebuffer *candidate, const Framebuffer *current) {
                    if (current == nullptr) {
                        return true;
                    }

                    if (candidate->lastWriteTimestamp != current->lastWriteTimestamp) {
                        return candidate->lastWriteTimestamp > current->lastWriteTimestamp;
                    }

                    const uint32_t candidateOffset = (candidate->addressStart > screenFbAddress) ? (candidate->addressStart - screenFbAddress) : (screenFbAddress - candidate->addressStart);
                    const uint32_t currentOffset = (current->addressStart > screenFbAddress) ? (current->addressStart - screenFbAddress) : (screenFbAddress - current->addressStart);
                    if (candidateOffset != currentOffset) {
                        return candidateOffset < currentOffset;
                    }

                    if (candidate->width != current->width) {
                        return candidate->width > current->width;
                    }

                    return candidate->height > current->height;
                };

                for (auto &fbEntry : fbManager.framebuffers) {
                    Framebuffer &candidate = fbEntry.second;
                    if (!candidate.contains(present.screenVI.origin, present.screenVI.origin + 1)) {
                        continue;
                    }

                    if ((candidate.siz != screenFbSiz) || (candidate.addressStart < storageBaseAddress)) {
                        continue;
                    }

                    const size_t storageOffset = size_t(candidate.addressStart - storageBaseAddress);
                    const size_t framebufferBytes = size_t(candidate.imageRowBytes(candidate.width)) * size_t(candidate.height);
                    if ((storageOffset >= present.storage.size()) || ((storageOffset + framebufferBytes) > present.storage.size())) {
                        continue;
                    }

                    if (isBetterOriginCandidate(&candidate, bestOriginFb)) {
                        bestOriginFb = &candidate;
                    }

                    if ((candidate.lastWriteType == Framebuffer::Type::Color) && (candidate.lastWriteFmt == G_IM_FMT_RGBA) &&
                        isBetterOriginCandidate(&candidate, bestColorOriginFb))
                    {
                        bestColorOriginFb = &candidate;
                    }
                }

                Framebuffer *originFb = (bestColorOriginFb != nullptr) ? bestColorOriginFb : bestOriginFb;
                if (originFb != nullptr) {
                    const size_t storageOffset = size_t(originFb->addressStart - storageBaseAddress);
                    directSourceFb = originFb;
                    directSourceOffset = uint32_t(storageOffset);
                    directSourceAddress = originFb->addressStart;
                    directSourceSize = { originFb->width, originFb->height };
                    directSourceSiz = originFb->siz;
                    directSourceTag = (originFb == bestColorOriginFb) ? "origin-color-fb" : "origin-fb";
                    directSourceType = originFb->lastWriteType;
                    directSourceFmt = originFb->lastWriteFmt;
                    directSourceBytes = present.storage.data() + storageOffset;
                    directSourceBytesCount = present.storage.size() - storageOffset;
                }
            }

            directSourceBaseBytes = directSourceBytes;
            directSourceBaseBytesCount = directSourceBytesCount;
            directSourceBaseOffset = directSourceOffset;
            directSourceBaseAddress = directSourceAddress;

#if defined(__ANDROID__)
            if ((directSourceSize.x > 0U) && (directSourceSize.y > 0U) && (directSourceSiz >= G_IM_SIZ_16b) &&
                (present.screenVI.origin > directSourceAddress))
            {
                const uint32_t rowBytes = uint32_t(directSourceSize.x) << (directSourceSiz - 1);
                const uint32_t originOffset = present.screenVI.origin - directSourceAddress;
                const size_t expectedBytes = size_t(rowBytes) * size_t(directSourceSize.y);
                const uint32_t originRows = (rowBytes > 0U) ? (originOffset / rowBytes) : 0U;
                const bool gameplayDirectSource = (directSourceFb != nullptr) && (std::strcmp(directSourceTag, "origin-color-fb") == 0) &&
                    (directSourceSize.x == 292U) && (directSourceSize.y == 216U) &&
                    ((directSourceBaseAddress == 0x003A5D00U) || (directSourceBaseAddress == 0x003C49C0U));
                if ((rowBytes > 0U) && ((originOffset % rowBytes) == 0U) &&
                    ((size_t(originOffset) + expectedBytes) <= directSourceBytesCount))
                {
                    if (gameplayDirectSource && (originRows > 0U)) {
                        const uint32_t directSourceHeight = uint32_t(directSourceSize.y);
                        directSourceSize.y = directSourceHeight + originRows;
                        directSourceBytes = directSourceBaseBytes;
                        directSourceBytesCount = directSourceBaseBytesCount;
                        directSourceOffset = directSourceBaseOffset;
                        directSourceAddress = directSourceBaseAddress;
                        directTextureCoordinateOffset.y = float(originRows) / float(uint32_t(directSourceSize.y));
                    }
                    else {
                        // The direct Android upload bypasses framebuffer history and VI origin handling, so crop
                        // the bytes to the visible VI origin row instead of uploading the extra lead-in row.
                        directSourceBytes += originOffset;
                        directSourceBytesCount -= originOffset;
                        directSourceOffset += originOffset;
                        directSourceAddress = present.screenVI.origin;
                    }
                }
            }

            const bool gameplayDirectSource = (directSourceFb != nullptr) && (std::strcmp(directSourceTag, "origin-color-fb") == 0) &&
                (directSourceSize.x == 292U) && ((directSourceSize.y == 216U) || (directSourceSize.y == 217U)) &&
                ((directSourceBaseAddress == 0x003A5D00U) || (directSourceBaseAddress == 0x003C49C0U));
            if (gameplayDirectSource && (g_android_gameplay_direct_vi_metadata_logs < 16U)) {
                const bool shouldLogGameplayMetadata = (directSourceAddress != g_android_gameplay_direct_vi_last_address) ||
                    (directSourceFb->lastWriteTimestamp != g_android_gameplay_direct_vi_last_timestamp);
                if (shouldLogGameplayMetadata) {
                    const uint32_t rowBytes = directSourceFb->imageRowBytes(directSourceFb->width);
                    const size_t expectedBytes = size_t(rowBytes) * size_t(directSourceFb->height);
                    const FixedRect &r = directSourceFb->lastWriteRect;
                    auto sampleContiguousRGBA16 = [&](uint32_t x, uint32_t y) -> uint16_t {
                        const size_t byteOffset = (size_t(y) * size_t(directSourceSize.x) + size_t(x)) * 2;
                        if ((byteOffset + 1) >= directSourceBytesCount) {
                            return 0;
                        }

                        return (uint16_t(directSourceBytes[byteOffset + 0]) << 8) | uint16_t(directSourceBytes[byteOffset + 1]);
                    };

                    auto sampleHalfwordRGBA16 = [&](uint32_t x, uint32_t y) -> uint16_t {
                        const size_t byteOffset = (size_t(y) * size_t(directSourceSize.x) + size_t(x)) * 2;
                        const size_t rawIndex = ((size_t(directSourceAddress) + byteOffset) ^ size_t(0x2U)) - size_t(directSourceAddress);
                        if ((rawIndex + 1) >= directSourceBytesCount) {
                            return 0;
                        }

                        return (uint16_t(directSourceBytes[rawIndex + 1]) << 8) | uint16_t(directSourceBytes[rawIndex + 0]);
                    };

                    auto rowStats = [&](uint32_t y, uint32_t &meanByte, uint32_t &nonzeroBytes) {
                        meanByte = 0;
                        nonzeroBytes = 0;
                        if (rowBytes == 0) {
                            return;
                        }

                        const size_t rowStart = size_t(y) * size_t(rowBytes);
                        const size_t rowEnd = rowStart + size_t(rowBytes);
                        if (rowEnd > directSourceBytesCount) {
                            return;
                        }

                        uint64_t sum = 0;
                        for (size_t i = rowStart; i < rowEnd; i++) {
                            const uint8_t v = directSourceBytes[i];
                            sum += v;
                            nonzeroBytes += (v != 0);
                        }

                        meanByte = uint32_t(sum / rowBytes);
                    };

                    const uint32_t directSourceWidth = uint32_t(directSourceSize.x);
                    const uint32_t directSourceHeight = uint32_t(directSourceSize.y);
                    const uint32_t centerX = directSourceWidth / 2U;
                    const uint32_t centerY = directSourceHeight / 2U;
                    const uint32_t lastX = (directSourceWidth > 0U) ? (directSourceWidth - 1U) : 0U;
                    const uint32_t lastY = (directSourceHeight > 0U) ? (directSourceHeight - 1U) : 0U;
                    uint32_t row0Mean = 0;
                    uint32_t row0Nonzero = 0;
                    uint32_t rowMidMean = 0;
                    uint32_t rowMidNonzero = 0;
                    uint32_t rowLastMean = 0;
                    uint32_t rowLastNonzero = 0;
                    rowStats(0, row0Mean, row0Nonzero);
                    rowStats(centerY, rowMidMean, rowMidNonzero);
                    rowStats(lastY, rowLastMean, rowLastNonzero);

                    BANJO_ANDROID_LOG("PresentQueue gameplay source meta: address=0x%08X end=0x%08X size=%ux%u readHeight=%u maxHeight=%u rowBytes=%u RAMBytes=%u storageBase=0x%08X storageOffset=0x%X storageBytes=%zu expectedBytes=%zu lastWriteTimestamp=%llu rect=(%d,%d)-(%d,%d) rectSize=%dx%d",
                        directSourceBaseAddress, directSourceFb->addressEnd, directSourceFb->width, directSourceFb->height, directSourceFb->readHeight, directSourceFb->maxHeight,
                        rowBytes, directSourceFb->RAMBytes, storageBaseAddress, directSourceOffset, directSourceBytesCount, expectedBytes,
                        static_cast<unsigned long long>(directSourceFb->lastWriteTimestamp), r.left(false), r.top(false), r.right(false), r.bottom(false),
                        r.width(false, false), r.height(false, false));
                    BANJO_ANDROID_LOG("PresentQueue gameplay source samples: rows[y0=%u/%u yMid=%u/%u yLast=%u/%u] contiguous=[%04X,%04X,%04X,%04X] halfword=[%04X,%04X,%04X,%04X]",
                        row0Mean, row0Nonzero, rowMidMean, rowMidNonzero, rowLastMean, rowLastNonzero,
                        sampleContiguousRGBA16(0, 0), sampleContiguousRGBA16(centerX, centerY), sampleContiguousRGBA16(lastX, centerY), sampleContiguousRGBA16(lastX, lastY),
                        sampleHalfwordRGBA16(0, 0), sampleHalfwordRGBA16(centerX, centerY), sampleHalfwordRGBA16(lastX, centerY), sampleHalfwordRGBA16(lastX, lastY));
                    if (g_android_gameplay_origin_candidate_logs < 4U) {
                        const uint32_t originOffset = (present.screenVI.origin >= directSourceBaseAddress) ? (present.screenVI.origin - directSourceBaseAddress) : (directSourceBaseAddress - present.screenVI.origin);
                        uint32_t candidateCount = 0;
                        BANJO_ANDROID_LOG("PresentQueue gameplay VI/source: viOrigin=0x%08X screenFb=0x%08X screenSize=%ux%u chosen=%s sourceAddress=0x%08X uploadAddress=0x%08X sourceSize=%ux%u uploadSize=%ux%u originOffset=0x%X",
                            present.screenVI.origin, screenFbAddress, uint32_t(screenFbSize.x), uint32_t(screenFbSize.y),
                            directSourceTag, directSourceBaseAddress, directSourceAddress, uint32_t(directSourceSize.x), uint32_t(directSourceSize.y), uint32_t(directSourceSize.x), uint32_t(directSourceSize.y), originOffset);
                        for (auto &fbEntry : fbManager.framebuffers) {
                            Framebuffer &candidate = fbEntry.second;
                            if (!candidate.contains(present.screenVI.origin, present.screenVI.origin + 1)) {
                                continue;
                            }

                            if ((candidate.siz != screenFbSiz) || (candidate.addressStart < storageBaseAddress)) {
                                continue;
                            }

                            const size_t candidateStorageOffset = size_t(candidate.addressStart - storageBaseAddress);
                            const size_t candidateFramebufferBytes = size_t(candidate.imageRowBytes(candidate.width)) * size_t(candidate.height);
                            if ((candidateStorageOffset >= present.storage.size()) || ((candidateStorageOffset + candidateFramebufferBytes) > present.storage.size())) {
                                continue;
                            }

                            const FixedRect &candidateRect = candidate.lastWriteRect;
                            BANJO_ANDROID_LOG("PresentQueue gameplay candidate[%u]: address=0x%08X end=0x%08X size=%ux%u type=%u fmt=%u timestamp=%llu readHeight=%u maxHeight=%u storageOffset=0x%zX rect=(%d,%d)-(%d,%d) rectSize=%dx%d",
                                candidateCount, candidate.addressStart, candidate.addressEnd, candidate.width, candidate.height, uint32_t(candidate.lastWriteType), uint32_t(candidate.lastWriteFmt),
                                static_cast<unsigned long long>(candidate.lastWriteTimestamp), candidate.readHeight, candidate.maxHeight, candidateStorageOffset,
                                candidateRect.left(false), candidateRect.top(false), candidateRect.right(false), candidateRect.bottom(false),
                                candidateRect.width(false, false), candidateRect.height(false, false));
                            candidateCount++;
                            if (candidateCount >= 8U) {
                                break;
                            }
                        }
                        BANJO_ANDROID_LOG("PresentQueue gameplay candidates total=%u", candidateCount);
                        g_android_gameplay_origin_candidate_logs++;
                    }

                    g_android_gameplay_direct_vi_metadata_logs++;
                    g_android_gameplay_direct_vi_last_address = directSourceAddress;
                    g_android_gameplay_direct_vi_last_timestamp = directSourceFb->lastWriteTimestamp;
                }
            }
#endif

            RenderTarget *uploadedChangesTarget = nullptr;
            RenderTarget *uploadedTarget = nullptr;
            if (colorTarget == nullptr) {
                Framebuffer *viFb = nullptr;
                if (!viewRDRAM) {
                    viFb = fbManager.find(screenFbAddress);
                }

                Framebuffer *presentFb = viFb;
                
                // Show the framebuffer the debugger has requested instead.
                if (present.debuggerFramebuffer.view) {
                    Framebuffer *candidateFb = fbManager.find(present.debuggerFramebuffer.address);
                    if (candidateFb != nullptr) {
                        presentFb = candidateFb;
                    }
                }
                
                if ((presentFb != nullptr) && (viFb != nullptr)) {
                    for (uint32_t colorAddress : ext.sharedResources->colorImageAddressVector) {
                        Framebuffer *colorFb = fbManager.find(colorAddress);
                        if (colorFb == nullptr) {
                            continue;
                        }

                        // Always default to interpolation being disabled for all modified framebuffers.
                        colorFb->interpolationEnabled = false;
                        
                        // When the skip buffering option is on, we check the video history to find if any of the framebuffers that
                        // were drawn in this frame have been previously used for presentation. This is ignored when the debugger
                        // has forced viewing a particular framebuffer.
                        if (!present.debuggerFramebuffer.view && (presentationMode == EnhancementConfiguration::Presentation::Mode::SkipBuffering)) {
                            for (size_t h = 0; h < viHistory.history.size(); h++) {
                                const VIHistory::Present &entry = viHistory.history[h];
                                if ((colorFb->addressStart == entry.vi.fbAddress()) && (colorFb->width == entry.fbWidth) && (colorFb->siz == entry.vi.fbSiz()) && entry.vi.compatibleWith(present.screenVI)) {
                                    presentFb = colorFb;
                                    break;
                                }
                            }
                        }

                        // Present early (or games that behave like it) will make it so that the presented image is a color image
                        // that the workload modified. We run a basic check to see if that holds true to indicate it was presented
                        // so interpolation is possible.
                        if (colorFb == presentFb) {
                            presentFb->interpolationEnabled = true;
                            break;
                        }
                    }

                    if (presentFb->interpolationEnabled) {
                        framesToPresent = frameCounters.count;
                    }
                    else {
                        lockedWorkloadMutex = true;
                        ext.sharedResources->workloadMutex.lock();
                    }

                    RenderTargetKey colorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, Framebuffer::Type::Color);
                    colorTarget = &targetManager.get(colorTargetKey, true);
                    if (!colorTarget->isEmpty()) {
                        // If a depth framebuffer is about to be shown, convert it to color.
                        if (presentFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                            const bool uploadedStorage = (presentFb->addressStart == screenFbAddress) && (presentFb->width == screenFbSize.x) &&
                                (presentFb->siz == screenFbSiz) && uploadPresentStorage(*colorTarget, screenStorageBytes, screenStorageBytesCount, screenFbAddress, screenFbSize, screenFbSiz, G_IM_FMT_RGBA);

                            if (uploadedStorage) {
                                framesToPresent = 1;
                            }
                            else {
                                RenderTargetKey otherColorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, presentFb->lastWriteType);
                                RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey, true);
                                if (!otherColorTarget.isEmpty()) {
                                    const FixedRect &r = presentFb->lastWriteRect;
                                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                                    colorTarget->copyFromTarget(ext.presentGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                                }
                            }
                        }
                    }
                    else if ((presentFb->addressStart == screenFbAddress) && (presentFb->width == screenFbSize.x) && (presentFb->siz == screenFbSiz) &&
                        uploadPresentStorage(*colorTarget, screenStorageBytes, screenStorageBytesCount, screenFbAddress, screenFbSize, screenFbSiz, G_IM_FMT_RGBA)) {
                        framesToPresent = 1;
                    }
                    else {
                        colorTarget = nullptr;
                    }

                    if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                        viHistory.pushVI(present.screenVI, viFb->width);
                    }
                }
                else {
                    RenderTargetKey colorTargetKey(screenFbAddress, screenFbSize.x, screenFbSiz, Framebuffer::Type::Color);
                    colorTarget = &targetManager.get(colorTargetKey, true);
                    if (!uploadPresentStorage(*colorTarget, screenStorageBytes, screenStorageBytesCount, screenFbAddress, screenFbSize, screenFbSiz, G_IM_FMT_RGBA)) {
                        colorTarget = nullptr;
                    }

                    if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                        viHistory.pushVI(present.screenVI, screenFbSize.x);
                    }
                }
            }

            if ((colorTarget == nullptr) && (uploadedChangesTarget == nullptr) && (uploadedTarget == nullptr)) {
                uploadedTarget = uploadPresentStorageDirect(directSourceBytes, directSourceBytesCount, directSourceAddress, directSourceSize, directSourceSiz, directSourceOffset, directSourceTag, directSourceType, directSourceFmt);
            }

            if ((colorTarget == nullptr) && (uploadedChangesTarget != nullptr)) {
                colorTarget = uploadedChangesTarget;
                framesToPresent = 1;

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, directSourceFb != nullptr ? directSourceFb->width : screenFbSize.x);
                }
            }
            else if ((colorTarget == nullptr) && (uploadedTarget != nullptr)) {
                colorTarget = uploadedTarget;
                framesToPresent = 1;
                renderTextureCoordinateOffset = directTextureCoordinateOffset;

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, directSourceFb != nullptr ? directSourceFb->width : screenFbSize.x);
                }
            }
        }

        // Create the framebuffers if necessary.
        if (swapChainFramebuffers.empty()) {
            uint32_t textureCount = ext.swapChain->getTextureCount();
            swapChainFramebuffers.resize(textureCount);
            for (uint32_t i = 0; i < textureCount; i++) {
                const RenderTexture *swapChainTexture = ext.swapChain->getTexture(i);
                swapChainFramebuffers[i] = ext.device->createFramebuffer(RenderFramebufferDesc(&swapChainTexture, 1));
            }

            static bool logged_swapchain_framebuffers = false;
            if (!logged_swapchain_framebuffers) {
                BANJO_ANDROID_LOG("PresentQueue created %u swapchain framebuffers", textureCount);
                logged_swapchain_framebuffers = true;
            }
        }
        
        for (int32_t i = 0; i < framesToPresent; i++) {
            uint32_t frameCountersNextPresented = 0;
            if ((framesToPresent > 1) && (usingMSAA || (i > 0))) {
                // Stall until the interpolated color target is available.
                const uint32_t targetIndex = usingMSAA ? i : (i - 1);
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                    return (frameCounters.available > targetIndex) || ((frameCounters.available == targetIndex) && frameCounters.skipped);
                });

                // Do not present any more frames after this one after reaching the last available frame if the workload was skipped.
                if ((frameCounters.available == targetIndex) && frameCounters.skipped) {
                    framesToPresent = std::min(int(frameCounters.available), i + 1);
                    frameCountersNextPresented = frameCounters.count;
                }
                else {
                    frameCountersNextPresented = frameCounters.presented + 1;
                }

                if (i < framesToPresent) {
                    uint32_t targetIndex = usingMSAA ? i : (i - 1);
                    colorTarget = ext.sharedResources->interpolatedColorTargets[targetIndex].get();
                }
                else {
                    colorTarget = nullptr;
                }
            }
            else if (framesToPresent == 1) {
                frameCountersNextPresented = frameCounters.count;
            }

            uint32_t swapChainIndex = 0;
            const bool presentFrame = (i < framesToPresent) && swapChainValid;
            if (presentFrame) {
                swapChainValid = ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex);
            }

            if (presentFrame && swapChainValid) {
                // Draw the framebuffer with the VI renderer.
                RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
                RenderFramebuffer *swapChainFramebuffer = swapChainFramebuffers[swapChainIndex].get();
                RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                commandList->begin();
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                
                VIRenderer::RenderParams renderParams;
                if (colorTarget != nullptr) {
                    renderParams.device = ext.device;
                    renderParams.commandList = commandList;
                    renderParams.swapChain = ext.swapChain;
                    renderParams.shaderLibrary = ext.shaderLibrary;
                    renderParams.textureFormat = colorTarget->format;
                    renderParams.resolutionScale = colorTarget->resolutionScale;
                    renderParams.downsamplingScale = 1;
                    renderParams.filtering = filtering;
                    renderParams.textureCoordinateOffset = renderTextureCoordinateOffset;
                    renderParams.vi = &present.screenVI;
                    renderParams.removeBlackBorders = removeBlackBorders;

                    const bool useDownsampling = (colorTarget->downsampleMultiplier > 1);
                    if (useDownsampling) {
                        colorTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->downsampledTexture.get();
                        renderParams.textureWidth = colorTarget->width / colorTarget->downsampleMultiplier;
                        renderParams.textureHeight = colorTarget->height / colorTarget->downsampleMultiplier;
                        renderParams.downsamplingScale = colorTarget->downsampleMultiplier;
                    }
                    else {
                        colorTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->getResolvedTexture();
                        renderParams.textureWidth = colorTarget->width;
                        renderParams.textureHeight = colorTarget->height;
                    }
                }
                
                commandList->setFramebuffer(swapChainFramebuffer);
                commandList->clearColor();

                RenderHookDraw *drawHook = GetRenderHookDraw();
                if (g_android_present_draw_logs < 64) {
                    BANJO_ANDROID_LOG("PresentQueue drawing frame[%u]: swapChainIndex=%u hasColorTarget=%d drawHook=%d",
                        g_android_present_draw_logs, swapChainIndex, renderParams.texture != nullptr, drawHook != nullptr);
                    g_android_present_draw_logs++;
                }

                if (renderParams.texture != nullptr) {
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(renderParams.texture, RenderTextureLayout::SHADER_READ));
                    viRenderer->render(renderParams);
                }

                if (drawHook != nullptr) {
                    drawHook(commandList, swapChainFramebuffer);
                }

                {
                    const std::scoped_lock lock(inspectorMutex);
                    if (inspector != nullptr) {
                        inspector->draw(commandList);
                    }
                    
                    commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::PRESENT));
                    commandList->end();
                    const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                    RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
                    RenderCommandSemaphore *signalSemaphore = drawSemaphores[swapChainIndex].get();
                    ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, &signalSemaphore, 1, ext.presentGraphicsWorker->commandFence.get());
                    ext.presentGraphicsWorker->wait();
                }
            }

            if (lockedWorkloadMutex) {
                ext.sharedResources->workloadMutex.unlock();
                lockedWorkloadMutex = false;
            }
            
            if (frameCountersNextPresented > 0) {
                {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    frameCounters.presented = frameCountersNextPresented;
                }

                ext.sharedResources->interpolatedCondition.notify_all();
            }

            // As soon as we're done with the first render target, we notify the workload queue it can proceed.
            if (i == 0) {
                notifyPresentId(present);
            }

            if (presentFrame && swapChainValid) {
                // Wait until the approximate time the next present should be at the current intended rate.
                if ((presentTimestamp != Timestamp()) && (targetRate > 0) && (targetRate > viOriginalRate)) {
                    Timer::preciseSleepUntil(presentTimestamp + std::chrono::nanoseconds(1'000'000'000 / targetRate));
                }

                if (presentWaitEnabled) {
                    ext.swapChain->wait();
                }

                RenderCommandSemaphore *waitSemaphore = drawSemaphores[swapChainIndex].get();
                presentTimestamp = Timer::current();
                swapChainValid = ext.swapChain->present(swapChainIndex, &waitSemaphore, 1);
                presentProfiler.logAndRestart();
            }
        }
    }

    void PresentQueue::skipInterpolation() {
        {
            std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
            InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
            frameCounters.presented = frameCounters.count;
        }

        ext.sharedResources->interpolatedCondition.notify_all();
    }

    void PresentQueue::notifyPresentId(const Present &present) {
        {
            std::scoped_lock<std::mutex> cursorLock(presentIdMutex);
            presentId = present.presentId;
        }

        presentIdCondition.notify_all();
    }
    
    void PresentQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % presents.size();
    }

    void PresentQueue::threadLoop() {
        Thread::setCurrentThreadName("RT64 Present");

        // Create the semaphore the acquire method will use.
        acquiredSemaphore = ext.device->createCommandSemaphore();

        // Create as many semaphores to signal as textures there are.
        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
        }

        // Since the swap chain might not need a resize right away, detect present wait.
#if defined(__ANDROID__)
        presentWaitEnabled = false;
#else
        presentWaitEnabled = ext.device->getCapabilities().presentWait;
#endif

        int processCursor = -1;
        bool skipPresent = false;
        uint32_t displayTimingRate = UINT32_MAX;
        const bool displayTiming = ext.device->getCapabilities().displayTiming;
        bool swapChainValid = !ext.swapChain->needsResize();
        while (presentThreadRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                cursorCondition.wait(cursorLock, [&]() {
                    return (writeCursor != threadCursor) || !presentThreadRunning;
                });

                if (presentThreadRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % presents.size();
#if defined(__ANDROID__)
                    skipPresent = false;
#else
                    skipPresent = (writeCursor != threadCursor);
#endif
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                const bool needsResize = ext.swapChain->needsResize() || !swapChainValid;
                if (needsResize) {
                    ext.presentGraphicsWorker->commandList->begin();
                    ext.presentGraphicsWorker->commandList->end();
                    ext.presentGraphicsWorker->execute();
                    ext.presentGraphicsWorker->wait();
                    swapChainValid = ext.swapChain->resize();
                    swapChainFramebuffers.clear();

                    BANJO_ANDROID_LOG("PresentQueue resize completed: valid=%d width=%u height=%u textures=%u",
                        swapChainValid, ext.swapChain->getWidth(), ext.swapChain->getHeight(), ext.swapChain->getTextureCount());

                    if (swapChainValid) {
                        ext.sharedResources->setSwapChainSize(ext.swapChain->getWidth(), ext.swapChain->getHeight());
                        
                        // Texture count could've changed after resize, so new semaphores are needed.
                        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
                            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
                        }
                    }
                }

                if (needsResize || ext.appWindow->detectWindowMoved()) {
                    ext.appWindow->detectRefreshRate();
                    ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), displayTimingRate));
                }

                if (displayTiming) {
                    uint32_t newDisplayTimingRate = ext.swapChain->getRefreshRate();
                    if (newDisplayTimingRate == 0) {
                        newDisplayTimingRate = UINT32_MAX;
                    }

                    if (newDisplayTimingRate != displayTimingRate) {
                        ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), newDisplayTimingRate));
                        displayTimingRate = newDisplayTimingRate;
                    }
                }

                skipPresent = skipPresent || ext.swapChain->isEmpty();
                static bool logged_skip_present = false;
                if (skipPresent && !logged_skip_present) {
                    BANJO_ANDROID_LOG("PresentQueue skipping present: swapChainEmpty=%d swapChainValid=%d",
                        ext.swapChain->isEmpty(), swapChainValid);
                    logged_skip_present = true;
                }

                Present &present = presents[processCursor];
                ext.workloadQueue->waitForWorkloadId(present.workloadId);

                if (!presentThreadRunning) {
                    continue;
                }

                if (skipPresent) {
                    skipInterpolation();
                    notifyPresentId(present);
                }
                else {
                    threadPresent(present, swapChainValid);
                }

                if (!present.paused) {
                    if (!present.fbOperations.empty()) {
                        const std::scoped_lock lock(screenFbChangePoolMutex);
                        screenFbChangePool.release(present.fbOperations.front().writeChanges.id);
                        present.fbOperations.clear();
                    }

                    threadAdvanceBarrier();
                }

                processCursor = -1;
            }
        }

        // Transition the active swap chain render target out of the present state to avoid live references to the resource.
        uint32_t swapChainIndex = 0;
        if (!ext.swapChain->isEmpty() && ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex)) {
            RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
            ext.presentGraphicsWorker->commandList->begin();
            ext.presentGraphicsWorker->commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
            ext.presentGraphicsWorker->commandList->end();

            const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
            RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
            ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, nullptr, 0, ext.presentGraphicsWorker->commandFence.get());
            ext.presentGraphicsWorker->wait();
        }
    }
};
