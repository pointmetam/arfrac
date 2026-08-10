//
//  arfaceTests.swift
//  arfaceTests
//
//  Created by lee on 2026/5/21.
//

import Testing
import CoreGraphics
import UIKit
@testable import arface

struct arfaceTests {

    @MainActor
    @Test func lipModePickerOnlyShowsCurrentComparisonModes() async throws {
        #expect(LipMakeupMode.visibleModes == [.arkitUVSegStudent, .lipRefinement])
    }

    @MainActor
    @Test func lipModeTitlesDistinguishModelPaths() async throws {
        #expect(LipMakeupMode.autoFusion.title == "自动融合")
        #expect(LipMakeupMode.arkit3D.title == "3D跟随")
        #expect(LipMakeupMode.lipRefinement.title == "候选模型")
        #expect(LipMakeupMode.lipRefinementLarge.title == "唇线B大")
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.title == "B+3D混合")
        #expect(LipMakeupMode.arkitUVStudent.title == "UV模型")
        #expect(LipMakeupMode.arkitUVSegStudent.title == "UV分区")
    }

    @MainActor
    @Test func lipModePickerTitlesFitCompactSegmentedControl() async throws {
        #expect(LipMakeupMode.autoFusion.pickerTitle == "自动")
        #expect(LipMakeupMode.arkit3D.pickerTitle == "3D")
        #expect(LipMakeupMode.lipRefinement.pickerTitle == "候选")
        #expect(LipMakeupMode.lipRefinementLarge.pickerTitle == "B大")
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.pickerTitle == "B3D")
        #expect(LipMakeupMode.arkitUVStudent.pickerTitle == "UV-A")
        #expect(LipMakeupMode.arkitUVSegStudent.pickerTitle == "UV-4C")
    }

    @MainActor
    @Test func lipModePerformanceLabelsDistinguishModelPaths() async throws {
        #expect(LipMakeupMode.autoFusion.performanceLabel == "Auto")
        #expect(LipMakeupMode.arkit3D.performanceLabel == nil)
        #expect(LipMakeupMode.lipRefinement.performanceLabel == "Lip ML 候选")
        #expect(LipMakeupMode.lipRefinementLarge.performanceLabel == "Lip ML B")
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.performanceLabel == "Lip ML B+3D")
        #expect(LipMakeupMode.arkitUVStudent.performanceLabel == "UV ML")
        #expect(LipMakeupMode.arkitUVSegStudent.performanceLabel == "UV 4C")
    }

    @MainActor
    @Test func hybrid3DLipModeUsesLargeModelWithScreenOverlay() async throws {
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.usesModel)
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.isLipRefinement)
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.usesLipROIModel)
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.usesLargeLipModel)
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.showsScreenModelOverlay)
        #expect(!LipMakeupMode.lipRefinementLarge3DGPU.usesGPULipOverlay)
        #expect(LipMakeupMode.lipRefinementLarge3DGPU.uses3DLipBase)
    }

    @MainActor
    @Test func autoFusionModeUsesBothLipModelsWithoutPermanentScreenOverlay() async throws {
        #expect(LipMakeupMode.autoFusion.usesModel)
        #expect(!LipMakeupMode.autoFusion.usesLipROIModel)
        #expect(!LipMakeupMode.autoFusion.usesUVLipStudent)
        #expect(!LipMakeupMode.autoFusion.showsScreenModelOverlay)
        #expect(!LipMakeupMode.autoFusion.uses3DLipBase)
    }

    @MainActor
    @Test func makeupStartsInUVSegStudentMode() async throws {
        let state = MakeupState()

        #expect(state.lipMode == .arkitUVSegStudent)
    }

    @Test func uvLipModeDoesNotUseStartup3DFallback() async throws {
        #expect(!LipMakeupMode.arkitUVSegStudent.usesStartup3DLipFallback(hasUVLipAlphaMask: false, isLipLineDebug: false))
        #expect(!LipMakeupMode.arkitUVSegStudent.usesStartup3DLipFallback(hasUVLipAlphaMask: true, isLipLineDebug: false))
        #expect(!LipMakeupMode.arkitUVSegStudent.usesStartup3DLipFallback(hasUVLipAlphaMask: false, isLipLineDebug: true))
        #expect(!LipMakeupMode.lipRefinement.usesStartup3DLipFallback(hasUVLipAlphaMask: false, isLipLineDebug: false))
    }

    @MainActor
    @Test func makeupLooksIncludeGreenLipDebugSwatch() async throws {
        let state = MakeupState()
        let debugLook = try #require(state.looks.first { $0.id == "debug-green-lip" })

        #expect(debugLook.shortName == "DBG-G")
        #expect(debugLook.lip.green > 0.85)
        #expect(debugLook.lip.alpha >= 0.90)
    }

    @Test func lipSoftAlphaDecoderKeepsProbabilityOutputAsAlpha() async throws {
        #expect(ARKitLipSoftAlphaDecoder.alphaByte(from: 0.0) == 0)
        #expect(ARKitLipSoftAlphaDecoder.alphaByte(from: 0.5) >= 127)
        #expect(ARKitLipSoftAlphaDecoder.alphaByte(from: 1.0) == 255)
    }

    @Test func lipSoftAlphaDecoderAcceptsRawLogitFallback() async throws {
        let low = ARKitLipSoftAlphaDecoder.alphaByte(from: -4)
        let high = ARKitLipSoftAlphaDecoder.alphaByte(from: 4)

        #expect(low > 0)
        #expect(high < 255)
        #expect(low < high)
    }

    @Test func lipTemplatePriorCreatesCenteredSoftLipShape() async throws {
        let width = 192
        let height = 128
        let template = ARKitLipTemplatePrior.canonical(width: width, height: height)

        #expect(template.count == width * height)
        #expect((template.min() ?? -1) >= 0)
        #expect((template.max() ?? 2) <= 1)

        let center = template[(height / 2) * width + (width / 2)]
        let upperCenter = template[(height * 40 / 100) * width + (width / 2)]
        let leftCorner = template[(height / 2) * width + (width * 15 / 100)]
        let topSkin = template[(height * 12 / 100) * width + (width / 2)]

        #expect(center > 0.85)
        #expect(upperCenter > 0.55)
        #expect(leftCorner < 0.20)
        #expect(topSkin < 0.05)
    }

    @MainActor
    @Test func makeupLooksIncludeSeparateLipLineDebugSwatches() async throws {
        let state = MakeupState()
        let fusedLine = try #require(state.looks.first { $0.id == "debug-lip-line" })
        let colorLine = try #require(state.looks.first { $0.id == "debug-lip-line-color" })
        let uvLine = try #require(state.looks.first { $0.id == "debug-lip-line-uv" })

        #expect(fusedLine.shortName == "LINE")
        #expect(fusedLine.isLipLineDebug)
        #expect(fusedLine.usesLipLineColorFusion)
        #expect(colorLine.shortName == "COLOR")
        #expect(colorLine.isLipLineDebug)
        #expect(colorLine.lipLineDebugSource == .colorOnly)
        #expect(!colorLine.usesLipLineColorFusion)
        #expect(uvLine.shortName == "UV")
        #expect(uvLine.isLipLineDebug)
        #expect(uvLine.lipLineDebugSource == .uvOnly)
        #expect(!uvLine.usesLipLineColorFusion)
    }

    @MainActor
    @Test func normalLookPickerHidesLipLineDebugSwatches() async throws {
        let state = MakeupState()

        #expect(state.visibleLooks.contains { $0.id == "debug-green-lip" })
        #expect(!state.visibleLooks.contains { $0.isLipLineDebug })
    }

    @MainActor
    @Test func lipLineDebugRenderDoesNotFallbackToPlainLipAlphaMask() async throws {
        let state = MakeupState()
        let lineLook = try #require(state.looks.first { $0.id == "debug-lip-line" })
        let lipAlpha = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 26, width: 28, height: 12))

        let image = UVMakeupTexture.render(
            look: lineLook,
            intensity: 1,
            faceGeometry: nil,
            lipAlphaMask: lipAlpha,
            lipLineMask: nil,
            drawsLips: true,
            drawsFaceMakeup: false,
            showsDebugGuides: false
        )

        #expect(maximumAlpha(in: image) == 0)
    }

    @Test func autoLipRenderDecisionUsesUVForNormalDistance() async throws {
        let decision = AutoLipRenderDecider.decide(
            mouthROIRatio: 0.28,
            uvMaskCoverage: 0.045,
            uvMaskMaximumAlpha: 220,
            hasFreshScreenMask: false,
            hasRecentScreenMask: false,
            uvModelReady: true,
            screenModelReady: true
        )

        #expect(decision == .uv4c)
    }

    @Test func autoLipRenderDecisionUsesScreenFallbackForCloseUp() async throws {
        let decision = AutoLipRenderDecider.decide(
            mouthROIRatio: 0.62,
            uvMaskCoverage: 0.045,
            uvMaskMaximumAlpha: 220,
            hasFreshScreenMask: true,
            hasRecentScreenMask: true,
            uvModelReady: true,
            screenModelReady: true
        )

        #expect(decision == .screen2D)
    }

    @Test func autoLipRenderDecisionFreezesRecentScreenMaskWhenCloseUpPredictionDrops() async throws {
        let decision = AutoLipRenderDecider.decide(
            mouthROIRatio: 0.62,
            uvMaskCoverage: 0.045,
            uvMaskMaximumAlpha: 220,
            hasFreshScreenMask: false,
            hasRecentScreenMask: true,
            uvModelReady: true,
            screenModelReady: true
        )

        #expect(decision == .freeze)
    }

    @Test func autoLipRenderDecisionKeepsUVForNormalDistanceEvenWhenUVMaskLooksBad() async throws {
        let decision = AutoLipRenderDecider.decide(
            mouthROIRatio: 0.38,
            uvMaskCoverage: 0.001,
            uvMaskMaximumAlpha: 12,
            hasFreshScreenMask: true,
            hasRecentScreenMask: true,
            uvModelReady: true,
            screenModelReady: true
        )

        #expect(decision == .uv4c)
    }

    @MainActor
    @Test func screenLipMaskTrackerScalesPreviousMaskToCloseUpROI() async throws {
        var tracker = ScreenLipMaskTracker(maxTrackingAge: 0.70)
        let initialROI = CGRect(x: 100, y: 220, width: 120, height: 80)
        let closeUpROI = CGRect(x: 70, y: 160, width: 240, height: 160)

        let acceptedFrame = tracker.acceptMaskFrame(initialROI, regionOfInterest: initialROI, now: 0)
        let trackedFrame = try #require(tracker.trackedFrame(for: closeUpROI, now: 0.08))

        #expect(acceptedFrame == initialROI)
        #expect(abs(trackedFrame.width - closeUpROI.width) < 0.001)
        #expect(abs(trackedFrame.height - closeUpROI.height) < 0.001)
        #expect(abs(trackedFrame.midX - closeUpROI.midX) < 0.001)
        #expect(abs(trackedFrame.midY - closeUpROI.midY) < 0.001)
    }

    @MainActor
    @Test func screenLipMaskTrackerKeepsRecentMaskAvailableForFreeze() async throws {
        var tracker = ScreenLipMaskTracker(maxTrackingAge: 0.70)
        let initialROI = CGRect(x: 100, y: 220, width: 120, height: 80)
        let nextROI = CGRect(x: 112, y: 224, width: 124, height: 82)

        tracker.acceptMaskFrame(initialROI, regionOfInterest: initialROI, now: 10)

        #expect(tracker.hasRecentMask(now: 10.35))
        #expect(tracker.trackedFrame(for: nextROI, now: 10.35) != nil)
    }

    @MainActor
    @Test func screenLipMaskTrackerExpiresOldMask() async throws {
        var tracker = ScreenLipMaskTracker(maxTrackingAge: 0.70)
        let initialROI = CGRect(x: 100, y: 220, width: 120, height: 80)
        let nextROI = CGRect(x: 112, y: 224, width: 124, height: 82)

        tracker.acceptMaskFrame(initialROI, regionOfInterest: initialROI, now: 20)

        #expect(!tracker.hasRecentMask(now: 21.0))
        #expect(tracker.trackedFrame(for: nextROI, now: 21.0) == nil)
    }

    @MainActor
    @Test func makeupIsHiddenWhileRecordingARKitUVSamples() async throws {
        let state = MakeupState()

        state.isMakeupVisible = true
        state.isRecordingARKitUV = true
        #expect(!state.isMakeupVisibleForARKit)

        state.isRecordingARKitUV = false
        #expect(state.isMakeupVisibleForARKit)
    }

    @MainActor
    @Test func makeupIsHiddenWhileRecordingLip2DSamples() async throws {
        let state = MakeupState()

        state.isMakeupVisible = true
        state.isRecordingLip2D = true

        #expect(state.isTrainingRecordingActive)
        #expect(!state.isMakeupVisibleForARKit)
    }

    @MainActor
    @Test func lip2DRecordingStateUsesPlainCameraRoute() async throws {
        let state = MakeupState()

        state.isRecordingLip2D = true

        #expect(state.usesLip2DCaptureView)
        #expect(state.trainingRecordingAccessibilityLabel == "提前停止高清 2D 唇部录制")
    }

    @MainActor
    @Test func lip2DRecordingDefaultsToBackCameraForSharpMouthSamples() async throws {
        let state = MakeupState()

        #expect(state.lip2DCameraPosition == .back)
    }

    @MainActor
    @Test func lip2DRecordingDisablesLiveLipDetection() async throws {
        let state = MakeupState()

        #expect(state.allowsLiveLipDetection)

        state.isRecordingLip2D = true

        #expect(!state.allowsLiveLipDetection)
    }

    @MainActor
    @Test func lipOnlyTestModeStartsEnabledForCleanComparisonPhotos() async throws {
        let state = MakeupState()

        #expect(state.isLipOnlyTestMode)
        #expect(state.photoCaptureRequestID == 0)
    }

    @Test func arkitUVRecordingGuideHasExpectedTotalDuration() async throws {
        #expect(ARKitUVRecordingGuide.default.totalDuration == 130)
    }

    @Test func arkitUVRecordingGuideShowsLeftTurnAtFortyTwoSeconds() async throws {
        let state = ARKitUVRecordingGuide.default.state(atElapsedSeconds: 42)

        #expect(state.phase.id == "left-turn")
        #expect(state.elapsedSeconds == 42)
        #expect(state.totalRemainingSeconds == 88)
        #expect(state.phaseRemainingSeconds == 13)
    }

    @Test func arkitUVRecordingGuideFinishesAfterFinalHold() async throws {
        let state = ARKitUVRecordingGuide.default.state(atElapsedSeconds: 130)

        #expect(state.isFinished)
        #expect(state.elapsedSeconds == 130)
        #expect(state.totalRemainingSeconds == 0)
    }

    @Test func uvLipAlphaLowerContourClampRemovesCentralDrip() async throws {
        let width = 32
        let height = 24
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 8...16 {
            for x in 7...24 {
                alpha[y * width + x] = 220
            }
        }
        for y in 17...21 {
            for x in 14...17 {
                alpha[y * width + x] = 220
            }
        }

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let clamped = mask.lowerContourClampedForLipLine()

        #expect(clamped.alpha[21 * width + 15] == 0)
        #expect(clamped.alpha[16 * width + 15] > 120)
        #expect(clamped.alpha[16 * width + 8] > 120)
    }

    @Test func uvLipAlphaLowerContourClampShrinksBroadLowerShelf() async throws {
        let width = 32
        let height = 24
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 8...20 {
            for x in 7...24 {
                alpha[y * width + x] = 220
            }
        }

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let clamped = mask.lowerContourClampedForLipLine(centerInsetRatio: 0.16, sideInsetRatio: 0.04, fadePixels: 4)

        #expect(clamped.alpha[20 * width + 15] == 0)
        #expect(clamped.alpha[17 * width + 15] > 80)
        #expect(clamped.alpha[20 * width + 8] > 0)
    }

    @MainActor
    @Test func uvLipEdgeRefinementShrinksAlphaAtOriginalBoundary() async throws {
        let refinement = UVLipEdgeRefinement.product

        let boundaryAlpha = refinement.normalizedAlpha(signedDistance: 0)
        let insideAlpha = refinement.normalizedAlpha(signedDistance: refinement.insetPixels + refinement.featherPixels + 0.5)

        #expect(boundaryAlpha < 0.35)
        #expect(insideAlpha > 0.84)
    }

    @MainActor
    @Test func uvLipEdgeRefinementKeepsOutsideFeatherSubtle() async throws {
        let refinement = UVLipEdgeRefinement.product

        let outsideAlpha = refinement.normalizedAlpha(signedDistance: -refinement.featherPixels * 0.5)

        #expect(outsideAlpha <= 0.08)
    }

    @Test func uvLipAlphaMaskTranslatedLeftMovesAlphaPixels() async throws {
        let width = 6
        let height = 4
        var alpha = [UInt8](repeating: 0, count: width * height)
        alpha[1 * width + 3] = 220
        alpha[2 * width + 0] = 180

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let translated = mask.translated(xPixels: -2, yPixels: 0)

        #expect(translated.alpha[1 * width + 1] == 220)
        #expect(translated.alpha[1 * width + 3] == 0)
        #expect(translated.alpha[2 * width + 0] == 0)
    }

    @Test func uvLipAlphaMaskProductShapePreservesUpperCenterByDefault() async throws {
        let width = 64
        let height = 40
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 10...24 {
            for x in 12...51 {
                alpha[y * width + x] = 220
            }
        }

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let shaped = mask.productLipSilhouetteRefined()

        #expect(shaped.alpha[10 * width + 32] > 120)
        #expect(shaped.alpha[11 * width + 32] > 120)
        #expect(shaped.alpha[10 * width + 23] > 120)
        #expect(shaped.alpha[10 * width + 41] > 120)
    }

    @Test func uvLipAlphaMaskProductShapeRoundsLowerSideShelf() async throws {
        let width = 64
        let height = 40
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 10...26 {
            for x in 12...51 {
                alpha[y * width + x] = 220
            }
        }

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let shaped = mask.productLipSilhouetteRefined()

        #expect(shaped.alpha[26 * width + 14] == 0)
        #expect(shaped.alpha[26 * width + 49] == 0)
        #expect(shaped.alpha[25 * width + 32] > 120)
    }

    @Test func uvLipMaskLockHoldsLockedMaskWhenCandidateOnlyJitters() async throws {
        var lock = UVLipMaskLock(stableFramesRequired: 2, relockFramesRequired: 3)
        let first = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 27, width: 24, height: 10))
        let second = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 21, y: 27, width: 24, height: 10))
        let jitter = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 28, width: 25, height: 10))

        _ = lock.resolve(candidate: first)
        let locked = lock.resolve(candidate: second)
        let held = lock.resolve(candidate: jitter)

        #expect(locked.decision == .locked)
        #expect(held.decision == .held)
        #expect(held.mask.alpha == locked.mask.alpha)
    }

    @Test func uvLipMaskLockCanFollowSimilarCandidateWhenConfigured() async throws {
        var lock = UVLipMaskLock(stableFramesRequired: 1, relockFramesRequired: 3)
        lock.holdUpdateResponse = 1
        let first = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 27, width: 24, height: 10))
        let follow = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 21, y: 27, width: 24, height: 10))

        _ = lock.resolve(candidate: first)
        let held = lock.resolve(candidate: follow)

        #expect(held.decision == .held)
        #expect(held.mask.alpha == follow.alpha)
    }

    @Test func uvLipMaskLockRejectsImplausibleJumpAndKeepsLockedMask() async throws {
        var lock = UVLipMaskLock(stableFramesRequired: 2, relockFramesRequired: 3)
        let first = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 27, width: 24, height: 10))
        let second = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 21, y: 27, width: 24, height: 10))
        let jump = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 3, y: 4, width: 10, height: 6))

        _ = lock.resolve(candidate: first)
        let locked = lock.resolve(candidate: second)
        let rejected = lock.resolve(candidate: jump)

        #expect(rejected.decision == .rejected)
        #expect(rejected.mask.alpha == locked.mask.alpha)
    }

    @Test func uvLipMaskLockRelocksAfterStablePlausibleShapeChange() async throws {
        var lock = UVLipMaskLock(stableFramesRequired: 2, relockFramesRequired: 3)
        let closedA = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 27, width: 24, height: 10))
        let closedB = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 21, y: 27, width: 24, height: 10))
        let openA = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 25, width: 28, height: 16))
        let openB = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 25, width: 29, height: 16))
        let openC = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 26, width: 29, height: 16))

        _ = lock.resolve(candidate: closedA)
        let locked = lock.resolve(candidate: closedB)
        let pending1 = lock.resolve(candidate: openA)
        let pending2 = lock.resolve(candidate: openB)
        let relocked = lock.resolve(candidate: openC)

        #expect(pending1.decision == .pendingRelock(1))
        #expect(pending1.mask.alpha == locked.mask.alpha)
        #expect(pending2.decision == .pendingRelock(2))
        #expect(relocked.decision == .relocked)
        #expect(relocked.mask.alpha == openC.alpha)
    }

    @Test func uvLipLineCandidateSelectorIgnoresColorSupportWhenDisabled() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 26, width: 28, height: 12), alphaValue: 220)
        let partialColor = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 28, y: 30, width: 10, height: 5), alphaValue: 230)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: partialColor,
            allowColorSupport: false
        )

        #expect(selected.mask.alpha == displayed.alpha)
        #expect(selected.status == " lineFuse uvOnly")
    }

    @Test func uvLipLineCandidateSelectorBlendsCredibleColorSupportWithModelMask() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        let colorSupport = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 26, width: 28, height: 12), alphaValue: 230)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: colorSupport,
            allowColorSupport: true
        )

        #expect(selected.status.contains(" lineFuse mid"))
        #expect(selected.mask.alpha[27 * 64 + 32] > displayed.alpha[27 * 64 + 32])
        #expect(selected.mask.alpha[27 * 64 + 32] < colorSupport.alpha[27 * 64 + 32])
        #expect(selected.mask.alpha[24 * 64 + 16] < displayed.alpha[24 * 64 + 16])
        #expect(selected.mask.alpha[24 * 64 + 16] > 0)
    }

    @Test func uvLipLineCandidateSelectorCanShowColorSupportOnly() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        let colorSupport = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 27, width: 24, height: 10), alphaValue: 230)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: colorSupport,
            source: .colorOnly
        )

        #expect(selected.mask.nonzeroCoverage < displayed.nonzeroCoverage)
        #expect(selected.mask.alpha[28 * 64 + 22] > 100)
        #expect(selected.mask.alpha[24 * 64 + 17] == 0)
        #expect(selected.status.contains("lineColor"))
    }

    @Test func uvLipLineGeometryGateAcceptsCenteredColorCandidate() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        let candidate = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 26, width: 28, height: 12), alphaValue: 230)

        let gate = UVLipLineGeometryGate.evaluate(reference: displayed, candidate: candidate)

        #expect(gate.passed)
        #expect(gate.cornerShiftRatio < 0.08)
        #expect(gate.centerShiftRatio < 0.04)
    }

    @Test func uvLipLineGeometryGateRejectsMouthCornerShift() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        let shifted = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 28, y: 26, width: 24, height: 12), alphaValue: 230)

        let gate = UVLipLineGeometryGate.evaluate(reference: displayed, candidate: shifted)

        #expect(!gate.passed)
        #expect(gate.cornerShiftRatio > 0.20)
    }

    @Test func uvLipLineCornerMirrorRepairUsesGoodCornerToTrimBadCorner() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        var colorAlpha = [UInt8](repeating: 0, count: 64 * 64)
        for y in 26...36 {
            for x in 18...45 {
                colorAlpha[y * 64 + x] = 230
            }
        }
        for y in 28...34 {
            for x in 8...17 {
                colorAlpha[y * 64 + x] = 230
            }
        }
        let colorSupport = UVLipAlphaMask(width: 64, height: 64, alpha: colorAlpha)

        let repaired = colorSupport.cornerMirrorRepaired(against: displayed)
        let gate = UVLipLineGeometryGate.evaluate(reference: displayed, candidate: repaired.mask)

        #expect(repaired.applied)
        #expect(repaired.side == "L")
        #expect(repaired.mask.alpha[30 * 64 + 10] == 0)
        #expect(repaired.mask.alpha[30 * 64 + 18] > 100)
        #expect(gate.passed)
    }

    @Test func uvLipLineCandidateSelectorFallsBackToUVWhenColorCornersShift() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        let shiftedColor = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 28, y: 26, width: 24, height: 12), alphaValue: 230)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: shiftedColor,
            source: .fused
        )

        #expect(selected.mask.alpha == displayed.alpha)
        #expect(selected.status.contains("anchorOff"))
    }

    @Test func uvLipLineCandidateSelectorRepairsSingleBadCornerBeforeFusion() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        var colorAlpha = [UInt8](repeating: 0, count: 64 * 64)
        for y in 26...36 {
            for x in 18...45 {
                colorAlpha[y * 64 + x] = 230
            }
        }
        for y in 28...34 {
            for x in 8...17 {
                colorAlpha[y * 64 + x] = 230
            }
        }
        let colorSupport = UVLipAlphaMask(width: 64, height: 64, alpha: colorAlpha)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: colorSupport,
            source: .fused
        )

        #expect(selected.status.contains("mirrorL"))
        #expect(selected.status.contains("lineFuse mid"))
        #expect(selected.mask.alpha[30 * 64 + 10] == 0)
        #expect(selected.mask.alpha[30 * 64 + 18] <= displayed.alpha[30 * 64 + 18] + 8)
    }

    @Test func uvLipLineFusionKeepsMouthCornersModelDominant() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        var colorAlpha = [UInt8](repeating: 0, count: 64 * 64)
        for y in 26...36 {
            for x in 18...45 {
                colorAlpha[y * 64 + x] = 230
            }
        }
        for y in 27...34 {
            for x in 10...17 {
                colorAlpha[y * 64 + x] = 230
            }
        }
        let colorSupport = UVLipAlphaMask(width: 64, height: 64, alpha: colorAlpha)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: colorSupport,
            allowColorSupport: true
        )

        #expect(selected.status.contains(" lineFuse mid"))
        #expect(selected.mask.alpha[30 * 64 + 32] > displayed.alpha[30 * 64 + 32])
        #expect(selected.mask.alpha[30 * 64 + 16] <= displayed.alpha[30 * 64 + 16] + 8)
    }

    @Test func uvLipCoverageGateAcceptsWellCoveredCandidate() async throws {
        let reference = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 24, width: 28, height: 14), alphaValue: 220)
        let candidate = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 19, y: 25, width: 26, height: 12), alphaValue: 220)

        let gate = UVLipCoverageGate.evaluate(reference: reference, candidate: candidate)

        #expect(gate.passed)
        #expect(gate.coverage > 0.72)
        #expect(gate.spillRate < 0.08)
        #expect(gate.areaRatio < 1.0)
    }

    @Test func uvLipCoverageGateRejectsLargeOuterSpill() async throws {
        let reference = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 22, y: 26, width: 20, height: 10), alphaValue: 220)
        let candidate = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 12, y: 24, width: 40, height: 14), alphaValue: 220)

        let gate = UVLipCoverageGate.evaluate(reference: reference, candidate: candidate)

        #expect(!gate.passed)
        #expect(gate.spillRate > 0.22)
        #expect(gate.areaRatio > 1.28)
    }

    @Test func uvLipLineCandidateSelectorRejectsTinyColorSupport() async throws {
        let displayed = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 16, y: 24, width: 32, height: 16), alphaValue: 180)
        let tinyColor = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 30, y: 30, width: 4, height: 3), alphaValue: 230)

        let selected = UVLipLineCandidateSelector.select(
            displayedMask: displayed,
            colorSupportMask: tinyColor,
            allowColorSupport: true
        )

        #expect(selected.mask.alpha == displayed.alpha)
        #expect(selected.status.contains(" lineFuse uvOnly"))
    }

    @Test func uvLipAlphaMaskFillsMissingAreaFromOverlappingLipLineMask() async throws {
        let base = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 28, width: 24, height: 8), alphaValue: 180)
        let lineFill = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 24, width: 28, height: 13), alphaValue: 220)

        let fused = base.filledByLipLine(lineFill, fillScale: 0.86, supportRadius: 5)

        #expect(fused.applied)
        #expect(fused.mask.alpha[25 * 64 + 32] > 120)
        #expect(fused.mask.alpha[29 * 64 + 32] >= base.alpha[29 * 64 + 32])
    }

    @Test func uvLipAlphaMaskRejectsFarLipLineFill() async throws {
        let base = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 28, width: 24, height: 8), alphaValue: 180)
        let farLineFill = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 3, y: 4, width: 10, height: 8), alphaValue: 220)

        let fused = base.filledByLipLine(farLineFill, fillScale: 0.86, supportRadius: 5)

        #expect(!fused.applied)
        #expect(fused.mask.alpha == base.alpha)
    }

    @Test func uvLipAlphaMaskSoftGateCanConservativelyRetainOutsideLinePixels() async throws {
        let base = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 24, width: 30, height: 14), alphaValue: 220)
        let lineGate = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 22, y: 26, width: 22, height: 10), alphaValue: 240)

        let gated = base.softlyGated(by: lineGate, minimumOverlapRatio: 0.30, outsideRetain: 0.58)

        #expect(gated.applied)
        #expect(gated.mask.alpha[30 * 64 + 32] > 200)
        #expect(gated.mask.alpha[24 * 64 + 19] > 100)
        #expect(gated.mask.alpha[24 * 64 + 19] < base.alpha[24 * 64 + 19])
    }

    @Test func uvLipAlphaMaskColorTrimReducesUnsupportedOuterPixels() async throws {
        let base = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 24, width: 28, height: 14), alphaValue: 220)
        let colorSupport = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 21, y: 26, width: 22, height: 10), alphaValue: 240)

        let trimmed = base.colorTrimmed(by: colorSupport, unsupportedRetain: 0.24, supportExpansionRadius: 1, blurRadius: 1)

        #expect(trimmed.applied)
        #expect(trimmed.mask.alpha[31 * 64 + 32] > 160)
        #expect(trimmed.mask.alpha[24 * 64 + 19] < base.alpha[24 * 64 + 19])
    }

    @Test func uvLipAlphaMaskColorTrimSkipsWeakSupport() async throws {
        let base = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 18, y: 24, width: 28, height: 14), alphaValue: 220)
        let tinySupport = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 30, y: 30, width: 3, height: 2), alphaValue: 240)

        let trimmed = base.colorTrimmed(by: tinySupport)

        #expect(!trimmed.applied)
        #expect(trimmed.mask.alpha == base.alpha)
    }

    @Test func uvLipAlphaMaskColorEdgeSnapKeepsBodyAndTrimsUnsupportedEdge() async throws {
        var alpha = [UInt8](repeating: 0, count: 64 * 64)
        for y in 22...39 {
            for x in 16...47 {
                let isEdge = x <= 18 || x >= 45 || y <= 24 || y >= 37
                alpha[y * 64 + x] = isEdge ? 112 : 226
            }
        }
        let base = UVLipAlphaMask(width: 64, height: 64, alpha: alpha)
        let colorSupport = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 25, width: 24, height: 12), alphaValue: 240)

        let snapped = base.colorEdgeSnapped(by: colorSupport, edgeAlphaThreshold: 150, unsupportedRetain: 0.18)

        #expect(snapped.applied)
        #expect(snapped.mask.alpha[30 * 64 + 32] == base.alpha[30 * 64 + 32])
        #expect(snapped.mask.alpha[23 * 64 + 17] < base.alpha[23 * 64 + 17])
    }

    @Test func uvLipAlphaMaskOutlineKeepsOnlyBoundaryPixels() async throws {
        let base = makeUVLipMask(width: 32, height: 32, rect: CGRect(x: 9, y: 11, width: 14, height: 8), alphaValue: 220)

        let outline = base.outline(threshold: 32, widthPixels: 1)

        #expect(outline.alpha[11 * 32 + 12] > 200)
        #expect(outline.alpha[15 * 32 + 16] == 0)
        #expect(outline.nonzeroCoverage < base.nonzeroCoverage * 0.55)
    }

    @Test func uvLipAlphaMaskOuterSilhouetteOutlineIgnoresInternalFragments() async throws {
        var alpha = [UInt8](repeating: 0, count: 48 * 32)
        for y in 10...20 {
            for x in 12...35 {
                alpha[y * 48 + x] = 220
            }
        }
        for y in 12...18 {
            for x in 16...31 where (x + y).isMultiple(of: 3) {
                alpha[y * 48 + x] = 0
            }
        }
        let fragmented = UVLipAlphaMask(width: 48, height: 32, alpha: alpha)

        let silhouette = fragmented.outerSilhouetteOutline(threshold: 32, widthPixels: 1)

        #expect(silhouette.alpha[10 * 48 + 20] > 200)
        #expect(silhouette.alpha[20 * 48 + 20] > 200)
        #expect(silhouette.alpha[15 * 48 + 24] == 0)
        #expect(silhouette.nonzeroCoverage < fragmented.nonzeroCoverage * 0.35)
    }

    @Test func uvLipAlphaMaskDebugLipLineOutlineRaisesLowerCentralDrag() async throws {
        var alpha = [UInt8](repeating: 0, count: 48 * 32)
        for y in 8...18 {
            for x in 12...35 {
                alpha[y * 48 + x] = 220
            }
        }
        for y in 19...24 {
            for x in 20...27 {
                alpha[y * 48 + x] = 220
            }
        }
        let dragged = UVLipAlphaMask(width: 48, height: 32, alpha: alpha)

        let rawOutline = dragged.outerSilhouetteOutline(threshold: 32, widthPixels: 1)
        let debugOutline = dragged.lipLineDebugOutline(threshold: 32, widthPixels: 1)

        #expect(rawOutline.alpha[24 * 48 + 24] > 200)
        #expect(debugOutline.alpha[24 * 48 + 24] == 0)
        #expect(debugOutline.alpha[22 * 48 + 24] > 200)
    }

    @Test func uvLipAlphaMaskSmoothLipLineSilhouetteClosesSmallBoundaryGaps() async throws {
        var alpha = [UInt8](repeating: 0, count: 48 * 32)
        for y in 10...20 {
            for x in 12...35 {
                alpha[y * 48 + x] = 220
            }
        }
        alpha[10 * 48 + 23] = 0
        alpha[10 * 48 + 24] = 0
        alpha[20 * 48 + 28] = 0
        alpha[20 * 48 + 29] = 0
        let notched = UVLipAlphaMask(width: 48, height: 32, alpha: alpha)

        let smoothed = notched.smoothLipLineSilhouette(threshold: 32, radius: 1)

        #expect(smoothed.alpha[10 * 48 + 23] > 180)
        #expect(smoothed.alpha[10 * 48 + 24] > 180)
        #expect(smoothed.alpha[20 * 48 + 28] > 180)
        #expect(smoothed.alpha[20 * 48 + 29] > 180)
        #expect(smoothed.alpha[6 * 48 + 23] == 0)
    }

    @Test func uvLipAlphaMaskConstrainedToLipRegionRemovesNoseLikeColorSupport() async throws {
        let lipRegion = makeUVLipMask(width: 64, height: 64, rect: CGRect(x: 20, y: 36, width: 24, height: 10), alphaValue: 220)
        var support = [UInt8](repeating: 0, count: 64 * 64)
        for y in 36...45 {
            for x in 20...43 {
                support[y * 64 + x] = 230
            }
        }
        for y in 16...24 {
            for x in 24...39 {
                support[y * 64 + x] = 230
            }
        }
        let colorSupport = UVLipAlphaMask(width: 64, height: 64, alpha: support)

        let constrained = colorSupport.constrainedToLipRegion(lipRegion, expansionRadius: 1)

        #expect(constrained.applied)
        #expect(constrained.mask.alpha[40 * 64 + 32] > 160)
        #expect(constrained.mask.alpha[20 * 64 + 32] == 0)
    }

    @Test func uvLipAlphaMaskSoftenedForMakeupAddsOuterFeatherWithoutChangingCore() async throws {
        let base = makeUVLipMask(width: 32, height: 32, rect: CGRect(x: 9, y: 11, width: 14, height: 8), alphaValue: 220)

        let softened = base.softenedForMakeup(edgeThreshold: 32, innerBlurRadius: 1, outerFeatherRadius: 3)

        #expect(softened.alpha[15 * 32 + 16] >= base.alpha[15 * 32 + 16])
        #expect(softened.alpha[10 * 32 + 16] > 0)
        #expect(softened.alpha[10 * 32 + 16] < softened.alpha[15 * 32 + 16])
        #expect(softened.alpha[7 * 32 + 16] == 0)
    }

    @Test func uvLipAlphaMaskSoftenedForMakeupSuppressesIsolatedBoundarySpike() async throws {
        var alpha = makeUVLipMask(width: 32, height: 32, rect: CGRect(x: 9, y: 11, width: 14, height: 8), alphaValue: 220).alpha
        alpha[8 * 32 + 16] = 220
        let spiked = UVLipAlphaMask(width: 32, height: 32, alpha: alpha)

        let softened = spiked.softenedForMakeup(edgeThreshold: 32, innerBlurRadius: 1, outerFeatherRadius: 3)

        #expect(softened.alpha[15 * 32 + 16] >= 220)
        #expect(softened.alpha[8 * 32 + 16] < 80)
    }

    @Test func uvLipAlphaMaskContourSmoothingKeepsMainLipAndRemovesSpeckles() async throws {
        var alpha = [UInt8](repeating: 0, count: 64 * 48)
        for y in 18...30 {
            for x in 16...47 {
                let edgeNotch = y == 18 && (x == 28 || x == 29 || x == 30)
                if !edgeNotch {
                    alpha[y * 64 + x] = 220
                }
            }
        }
        alpha[8 * 64 + 30] = 220
        alpha[9 * 64 + 31] = 220
        alpha[40 * 64 + 12] = 220
        let noisy = UVLipAlphaMask(width: 64, height: 48, alpha: alpha)

        let smoothed = noisy.productContourSmoothedForMakeup(edgeThreshold: 32, edgeFeatherPixels: 0)

        #expect(smoothed.alpha[24 * 64 + 32] > 200)
        #expect(smoothed.alpha[18 * 64 + 29] < 80)
        #expect(smoothed.alpha[8 * 64 + 30] == 0)
        #expect(smoothed.alpha[40 * 64 + 12] == 0)
    }

    @Test func uvLipAlphaMaskContourFilteringPreservesOriginalTopology() async throws {
        var alpha = [UInt8](repeating: 0, count: 96 * 64)
        for y in 24...41 {
            for x in 20...75 {
                let upperNotch = y <= 26 && (x >= 43 && x <= 52)
                let leftCornerCut = x < 24 && y < 30
                let rightCornerCut = x > 71 && y < 30
                if !upperNotch && !leftCornerCut && !rightCornerCut {
                    alpha[y * 96 + x] = 220
                }
            }
        }
        let blocky = UVLipAlphaMask(width: 96, height: 64, alpha: alpha)

        let smoothed = blocky.productContourSmoothedForMakeup(edgeThreshold: 32, edgeFeatherPixels: 0)

        #expect(smoothed.alpha[33 * 96 + 48] > 200)
        #expect(smoothed.alpha[23 * 96 + 48] == 0)
        #expect(smoothed.alpha[26 * 96 + 48] < 80)
        #expect(smoothed.alpha[24 * 96 + 20] == 0)
        #expect(smoothed.alpha[40 * 96 + 20] > 200)
    }

    @Test func uvLipAlphaMaskContourSmoothingPreservesAsymmetricLipShape() async throws {
        let width = 96
        let height = 64
        var alpha = [UInt8](repeating: 0, count: width * height)
        for x in 16...79 {
            let t = CGFloat(x - 16) / CGFloat(79 - 16)
            let upperCurve = Int((22 + 5 * abs(t - 0.50) + (t > 0.68 ? 2 : 0)).rounded())
            let lowerCurve = Int((40 - 4 * abs(t - 0.56) + (t < 0.30 ? 2 : 0)).rounded())
            for y in upperCurve...lowerCurve {
                alpha[y * width + x] = 220
            }
        }
        let shaped = UVLipAlphaMask(width: width, height: height, alpha: alpha)

        let smoothed = shaped.productContourSmoothedForMakeup(edgeThreshold: 32, edgeFeatherPixels: 0)

        #expect(smoothed.alpha[31 * width + 48] > 200)
        #expect(smoothed.alpha[22 * width + 48] > 80)
        #expect(smoothed.alpha[40 * width + 23] > 80)
        #expect(smoothed.alpha[20 * width + 48] == 0)
        #expect(smoothed.alpha[43 * width + 73] == 0)
    }

    @Test func uvLipAlphaMaskContourSmoothingPreservesInnerMouthGap() async throws {
        let width = 96
        let height = 64
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 22...28 {
            for x in 20...75 {
                alpha[y * width + x] = 220
            }
        }
        for y in 35...42 {
            for x in 22...73 {
                alpha[y * width + x] = 220
            }
        }
        let separatedLips = UVLipAlphaMask(width: width, height: height, alpha: alpha)

        let smoothed = separatedLips.productContourSmoothedForMakeup(edgeThreshold: 32, edgeFeatherPixels: 0)

        #expect(smoothed.alpha[25 * width + 48] > 200)
        #expect(smoothed.alpha[38 * width + 48] > 200)
        #expect(smoothed.alpha[32 * width + 48] == 0)
    }

    @Test func uvLipAlphaMaskContourSmoothingSupersamplesEdgesWithoutFillingMouthGap() async throws {
        let width = 32
        let height = 24
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 7...10 {
            for x in 8...23 {
                alpha[y * width + x] = 230
            }
        }
        for y in 14...17 {
            for x in 9...22 {
                alpha[y * width + x] = 230
            }
        }
        let separatedLips = UVLipAlphaMask(width: width, height: height, alpha: alpha)

        let smoothed = separatedLips.productContourSmoothedForMakeup(
            edgeThreshold: 32,
            edgeFeatherPixels: 0,
            outputScale: 3
        )

        #expect(smoothed.width == width * 3)
        #expect(smoothed.height == height * 3)
        #expect(smoothed.alpha[(9 * 3) * smoothed.width + (16 * 3)] > 180)
        #expect(smoothed.alpha[(12 * 3) * smoothed.width + (16 * 3)] == 0)
        #expect(smoothed.alpha.contains { $0 > 0 && $0 < 230 })
    }

    @Test func uvLipAlphaMaskContourSmoothingDoesNotBridgeThinMouthSlit() async throws {
        let width = 64
        let height = 44
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 13...19 {
            for x in 13...50 {
                alpha[y * width + x] = 230
            }
        }
        for y in 23...31 {
            for x in 15...48 {
                alpha[y * width + x] = 230
            }
        }
        for y in 18...25 {
            alpha[y * width + 13] = 230
            alpha[y * width + 50] = 230
        }
        let connectedLipShape = UVLipAlphaMask(width: width, height: height, alpha: alpha)

        let smoothed = connectedLipShape.productContourSmoothedForMakeup(
            edgeThreshold: 32,
            smoothingIterations: 3,
            edgeFeatherPixels: 0,
            outputScale: 3
        )

        #expect(smoothed.alpha[(16 * 3) * smoothed.width + (32 * 3)] > 180)
        #expect(smoothed.alpha[(27 * 3) * smoothed.width + (32 * 3)] > 180)
        #expect(smoothed.alpha[(21 * 3) * smoothed.width + (32 * 3)] == 0)
    }

    @Test func arkitLipMaskDisplayMaskRestoresCreaseAndSmoothsEdges() async throws {
        let width = 32
        let height = 20
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 5...15 {
            for x in 6...25 {
                alpha[y * width + x] = 220
            }
        }
        for x in 8...23 {
            alpha[10 * width + x] = 92
        }

        let mask = ARKitLipMask(
            frame: CGRect(x: 0, y: 0, width: width, height: height),
            width: width,
            height: height,
            alpha: alpha
        )

        let display = mask.displayMask(outputScale: 3)

        #expect(display.width == width * 3)
        #expect(display.height == height * 3)
        #expect(display.alpha[(8 * 3) * display.width + (16 * 3)] > 180)
        #expect(display.alpha[(13 * 3) * display.width + (16 * 3)] > 180)
        #expect(display.alpha[(10 * 3) * display.width + (16 * 3)] < 24)
        #expect(display.alpha.contains { $0 > 0 && $0 < 220 })
    }

    @Test func uvLipAlphaMaskPreservedCreaseClearsFullGap() async throws {
        let width = 32
        let height = 20
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 5...15 {
            for x in 6...25 {
                alpha[y * width + x] = 220
            }
        }

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let result = mask.preservedClosedMouthCrease()

        let midY = 10
        #expect(result.alpha[midY * width + 16] == 0)
        #expect(result.alpha[(midY - 1) * width + 16] == 0)
        #expect(result.alpha[(midY + 1) * width + 16] == 0)
        #expect(result.alpha[7 * width + 16] == 220)
        #expect(result.alpha[13 * width + 16] == 220)
    }

    @Test func uvLipAlphaMaskPreservedClosedMouthCreaseCutsCenterSeam() async throws {
        let width = 32
        let height = 20
        var alpha = [UInt8](repeating: 0, count: width * height)
        for y in 5...15 {
            for x in 6...25 {
                alpha[y * width + x] = 220
            }
        }
        for x in 8...23 {
            alpha[10 * width + x] = 92
        }

        let mask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let creased = mask.preservedClosedMouthCrease()

        #expect(creased.alpha[8 * width + 16] > 180)
        #expect(creased.alpha[13 * width + 16] > 180)
        #expect(creased.alpha[10 * width + 16] < 24)
    }

    private func makeUVLipMask(width: Int, height: Int, rect: CGRect, alphaValue: UInt8 = 220) -> UVLipAlphaMask {
        var alpha = [UInt8](repeating: 0, count: width * height)
        let minX = max(0, Int(rect.minX.rounded(.down)))
        let maxX = min(width - 1, Int(rect.maxX.rounded(.up)))
        let minY = max(0, Int(rect.minY.rounded(.down)))
        let maxY = min(height - 1, Int(rect.maxY.rounded(.up)))
        guard minX <= maxX, minY <= maxY else {
            return UVLipAlphaMask(width: width, height: height, alpha: alpha)
        }

        for y in minY...maxY {
            for x in minX...maxX {
                alpha[y * width + x] = alphaValue
            }
        }
        return UVLipAlphaMask(width: width, height: height, alpha: alpha)
    }

    private func maximumAlpha(in image: UIImage) -> UInt8 {
        guard let cgImage = image.cgImage else { return 0 }

        let width = cgImage.width
        let height = cgImage.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return 0
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return stride(from: 3, to: rgba.count, by: 4).reduce(UInt8(0)) { max($0, rgba[$1]) }
    }

}
