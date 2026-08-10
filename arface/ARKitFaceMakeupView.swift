//
//  ARKitFaceMakeupView.swift
//  arface
//
//  Created by lee on 2026/5/25.
//

import ARKit
import SceneKit
import SwiftUI
import UIKit

struct ARKitFaceMakeupView: UIViewRepresentable {
    @ObservedObject var state: MakeupState

    func makeUIView(context: Context) -> ARKitFaceMakeupContainerView {
        let view = ARKitFaceMakeupContainerView()
        view.onStatusChange = { [weak state] status in
            Task { @MainActor in
                guard state?.status != status else { return }
                state?.status = status
            }
        }
        view.onPerformanceChange = { [weak state] text in
            Task { @MainActor in
                guard state?.performanceText != text else { return }
                state?.performanceText = text
            }
        }
        view.onRecordingStatusChange = { [weak state] text in
            Task { @MainActor in
                state?.recordingText = text
            }
        }
        view.onRecordingActiveChange = { [weak state] active in
            Task { @MainActor in
                guard state?.isRecordingARKitUV != active else { return }
                state?.isRecordingARKitUV = active
            }
        }
        view.onPhotoCaptureStatusChange = { [weak state] text in
            Task { @MainActor in
                state?.photoText = text
            }
        }
        view.updateStyle(
            look: state.selectedLook,
            intensity: state.intensity,
            makeupVisible: state.isMakeupVisibleForARKit,
            lipMode: state.lipMode,
            showFaceMesh: state.showFaceMesh,
            lipOnlyTestMode: state.isLipOnlyTestMode,
            recordARKitUVSamples: state.isRecordingARKitUV,
            forceRawUVOnly: state.forceRawUVOnly
        )
        view.capturePhotoIfNeeded(requestID: state.photoCaptureRequestID)
        view.startIfPossible()
        return view
    }

    func updateUIView(_ view: ARKitFaceMakeupContainerView, context: Context) {
        view.updateStyle(
            look: state.selectedLook,
            intensity: state.intensity,
            makeupVisible: state.isMakeupVisibleForARKit,
            lipMode: state.lipMode,
            showFaceMesh: state.showFaceMesh,
            lipOnlyTestMode: state.isLipOnlyTestMode,
            recordARKitUVSamples: state.isRecordingARKitUV,
            forceRawUVOnly: state.forceRawUVOnly
        )
        view.capturePhotoIfNeeded(requestID: state.photoCaptureRequestID)
        view.startIfPossible()
    }

    static func dismantleUIView(_ view: ARKitFaceMakeupContainerView, coordinator: ()) {
        view.pause()
    }
}

final class ARKitFaceMakeupContainerView: UIView, ARSCNViewDelegate, ARSessionDelegate {
    var onStatusChange: ((MakeupTrackingStatus) -> Void)?
    var onPerformanceChange: ((String) -> Void)?
    var onRecordingStatusChange: ((String) -> Void)?
    var onRecordingActiveChange: ((Bool) -> Void)?
    var onPhotoCaptureStatusChange: ((String) -> Void)?

    private let sceneView = ARSCNView(frame: .zero)
    private let material = SCNMaterial()
    private let lipOverlayMaterial = SCNMaterial()
    private let preciseLipMakeupLayer = CALayer()
    private let preciseLipMaskLayer = CALayer()
    private let lipParsingOverlayLayer = CALayer()
    // Only the default UV path loads at view startup. Old experiment paths load on demand.
    private lazy var lipParsingModel = ARKitLipParsingModel()
    private lazy var lipRefinementSmallModel = ARKitLipRefinementModel(resourceName: "LipTemplatePrior192x128")
    private lazy var lipRefinementLargeModel = ARKitLipRefinementModel(resourceName: "LipParsingLarge")
    private lazy var lipSDFRefinementModel = ARKitLipRefinementModel(resourceName: "LipSDFStudent384", outputKind: .sdfAlpha)
    private lazy var lipLineAlphaModel = ARKitLipRefinementModel(resourceName: "LipAlphaStudent256", outputKind: .softAlpha)
    private lazy var lipCurveModel = ARKitLipCurveModel(resourceName: "LipCurveStudent384")
    private lazy var uvLipAlphaStudentModel = ARKitUVLipStudentModel(resourceName: "ARKitUVLipAlphaBest")
    private lazy var uvLipSegStudentModel = ARKitUVLipStudentModel(resourceName: "ARKitUVLipParsingSmall")
    private lazy var rawUVHotStartModel = ARKitUVLipStudentModel(resourceName: "ARKitUVLipParsingSmall")
    private let lipParsingQueue = DispatchQueue(label: "com.perfectmakeup.arkit.lip-parsing", qos: .utility)
    private let sdfLipParsingQueue = DispatchQueue(label: "com.perfectmakeup.arkit.sdf-lip-parsing", qos: .userInitiated)
    private let rawUVHotStartQueue = DispatchQueue(label: "com.perfectmakeup.arkit.raw-uv-hot-start", qos: .userInitiated)
    private let sdfHotStartQueue = DispatchQueue(label: "com.perfectmakeup.arkit.sdf-hot-start", qos: .userInitiated)
    private let startupModelWarmupQueue = DispatchQueue(label: "com.perfectmakeup.arkit.startup-warmup", qos: .userInitiated)
    private let auxiliaryModelWarmupQueue = DispatchQueue(label: "com.perfectmakeup.arkit.aux-warmup", qos: .utility)
    private let textureRenderQueue = DispatchQueue(label: "com.perfectmakeup.arkit.texture-render", qos: .userInitiated)
    private let faceParsingCompositor = FaceParsingMetalCompositor()
    private let lipMeshProjector = LipMaskFaceMeshProjector()
    private let uvLipMaskFeather = UVLipMaskMetalFeather()
    private let enableUVLipLineAssist = true
    private var uvLipLineAssistWarmupStarted = false
    private var uvLipLineAssistReady = false
    private let uvSegLipMaskHorizontalOffsetPixels = -2
    private let uvSampleRecorder = ARKitUVSampleRecorder()
    private let facePoseNode = SCNNode()
    private var currentLook: MakeupLook?
    private var currentIntensity = 0.72
    private var currentMakeupVisible = true
    private var currentLipMode: LipMakeupMode = .arkitUVSegStudent
    private var currentForceRawUVOnly = false
    private var activeAutoLipRenderPath: AutoLipRenderPath = .uv4c
    private var currentShowFaceMesh = false
    private var currentLipOnlyTestMode = true
    private var currentRecordARKitUVSamples = false
    private var hasStarted = false
    private var needsTextureRefresh = true
    private var textureRenderGeneration = 0
    private var isTextureRenderInFlight = false
    private var lipParsingGeneration = 0
    private var updateFrameCount = 0
    private var isLipParsingInFlight = false
    private var isRawUVHotStartInFlight = false
    private var isSDFHotStartInFlight = false
    private var lastLipParsingStartedAt: CFTimeInterval = 0
    private var latestFaceParsingMask: ARKitFaceParsingMask?
    private var latestFaceUVMask: UVFaceParsingMask?
    private var latestUVLipAlphaMask: UVLipAlphaMask?
    private var latestUVLipSmoothingSourceMask: UVLipAlphaMask?
    private var latestUVLipColorSupportMask: UVLipAlphaMask?
    private var latestUVLipLineRenderMask: UVLipAlphaMask?
    private var latestUVLipLineDebugMask: UVLipAlphaMask?
    private var latestUVLipLineFusedDebugMask: UVLipAlphaMask?
    private var latestUVLipLineOnlyDebugMask: UVLipAlphaMask?
    private var latestUVLipLineColorDebugMask: UVLipAlphaMask?
    private var latestCurveAssistMask: UVLipAlphaMask?
    private var uvLipMaskLock = ARKitFaceMakeupContainerView.makeUVLipMaskLock()
    private var uvLipLineFusedMaskLock = ARKitFaceMakeupContainerView.makeUVLipLineMaskLock()
    private var uvLipLineOnlyMaskLock = ARKitFaceMakeupContainerView.makeUVLipLineMaskLock()
    private var uvLipLineColorMaskLock = ARKitFaceMakeupContainerView.makeUVLipLineMaskLock()
    private var didResetUVLipMaskForCurveOuterV2 = false
    private var mainLipModelsWarmupStarted = false
    private var latestMouthOpenAmount: CGFloat = 0
    private var curveAssistFrameCounter = 0
    private var curveRejectStreak = 0
    private var curveBackoffFramesRemaining = 0
    private var usesForcedRawUVPipeline: Bool {
        currentForceRawUVOnly || currentLook?.id == "debug-green-lip"
    }
    private static func makeUVLipMaskLock() -> UVLipMaskLock {
        var lock = UVLipMaskLock(stableFramesRequired: 4, relockFramesRequired: 2)
        lock.stableIOU = 0.58
        lock.stableCoverageRatio = 0.64
        lock.stableCenterShiftPixels = 14
        lock.holdIOU = 0.68
        lock.holdCoverageRatio = 0.68
        lock.holdCenterShiftPixels = 14
        lock.relockIOU = 0.34
        lock.relockCoverageRatio = 0.48
        lock.relockCenterShiftPixels = 24
        lock.holdUpdateResponse = 0.30
        return lock
    }
    private static func makeUVLipLineMaskLock() -> UVLipMaskLock {
        var lock = UVLipMaskLock(stableFramesRequired: 3, relockFramesRequired: 3)
        lock.stableIOU = 0.54
        lock.stableCoverageRatio = 0.58
        lock.stableCenterShiftPixels = 16
        lock.holdIOU = 0.56
        lock.holdCoverageRatio = 0.60
        lock.holdCenterShiftPixels = 18
        lock.relockIOU = 0.40
        lock.relockCoverageRatio = 0.54
        lock.relockCenterShiftPixels = 20
        lock.holdUpdateResponse = 0.24
        return lock
    }

    private func emitPipelineConsoleLog(trigger: String) {
        let summary = latestLipParsingBreakdownText ?? "no-breakdown"
        let line = "PIPE trigger=\(trigger) mode=\(currentLipMode.rawValue) look=\(currentLook?.name ?? currentLook?.id ?? "nil") \(summary)"
        guard line != lastPipelineConsoleLine else { return }
        lastPipelineConsoleLine = line
        print(line)
    }

    private func emitPipelineConsoleLog(trigger: String, summary: String) {
        let line = "PIPE trigger=\(trigger) mode=\(currentLipMode.rawValue) look=\(currentLook?.name ?? currentLook?.id ?? "nil") \(summary)"
        guard line != lastPipelineConsoleLine else { return }
        lastPipelineConsoleLine = line
        print(line)
    }

    private func resetTryOnStartupTrace(reason: String) {
        startupTraceID += 1
        startupTraceStartedAt = CACurrentMediaTime()
        didLogStartupFaceAnchor = false
        didLogStartupUVRequest = false
        didLogStartupHotStart = false
        didLogStartupTextureApplied = false
        hasRenderedRawUVUltraFastFirstFrame = false
        logTryOnStartupEvent(stage: "reset", details: reason)
    }

    private func logTryOnStartupEvent(stage: String, details: String) {
        let elapsedMS: Double
        if let startupTraceStartedAt {
            elapsedMS = (CACurrentMediaTime() - startupTraceStartedAt) * 1000
        } else {
            elapsedMS = -1
        }
        print(
            String(
                format: "TRYON-START id=%d stage=%@ t=%.0fms mode=%@ raw=%d look=%@ %@",
                startupTraceID,
                stage,
                elapsedMS,
                currentLipMode.rawValue,
                usesForcedRawUVPipeline ? 1 : 0,
                currentLook?.id ?? "nil",
                details
            )
        )
    }

    private func emitMainThreadTimingLog(
        stage: String,
        totalMS: Double,
        thresholdMS: Double,
        details: String
    ) {
        guard currentLipMode == .lipSDFRefinement, totalMS >= thresholdMS else { return }
        let now = CACurrentMediaTime()
        guard now - lastMainThreadTimingLogAt > 0.18 else { return }
        lastMainThreadTimingLogAt = now
        print(
            String(
                format: "PIPE-MAIN stage=%@ total=%.1fms mode=%@ %@",
                stage,
                totalMS,
                currentLipMode.rawValue,
                details
            )
        )
    }

    private func lipMaskMetrics(_ mask: UVLipAlphaMask, threshold: UInt8 = 32) -> (coverage: CGFloat, center: CGPoint)? {
        guard mask.width > 0, mask.height > 0, mask.alpha.count == mask.width * mask.height else {
            return nil
        }

        var active = 0
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask.alpha[(y * mask.width) + x] > threshold {
                active += 1
                sumX += CGFloat(x)
                sumY += CGFloat(y)
            }
        }

        guard active > 8 else { return nil }
        let count = CGFloat(active)
        return (
            coverage: count / CGFloat(max(mask.alpha.count, 1)),
            center: CGPoint(x: sumX / count, y: sumY / count)
        )
    }

    private func stabilizeRawUVArea(reference: UVLipAlphaMask, candidate: UVLipAlphaMask) -> UVLipAlphaMask {
        guard
            let referenceMetrics = lipMaskMetrics(reference),
            let candidateMetrics = lipMaskMetrics(candidate)
        else {
            return reference
        }

        func stabilizedOffset(_ delta: CGFloat, response: CGFloat, deadZone: CGFloat) -> Int {
            guard abs(delta) > deadZone else { return 0 }

            let adjusted = delta.sign == .minus
                ? -(abs(delta) - deadZone)
                : (abs(delta) - deadZone)
            return Int(round(adjusted * response))
        }

        // Keep the locked lip area stable, but still allow the whole mask to follow
        // the mouth with a softer response so tiny center jitter does not show up.
        let deltaX = stabilizedOffset(
            candidateMetrics.center.x - referenceMetrics.center.x,
            response: 0.58,
            deadZone: 1.2
        )
        let deltaY = stabilizedOffset(
            candidateMetrics.center.y - referenceMetrics.center.y,
            response: 0.44,
            deadZone: 1.0
        )
        return reference.translated(xPixels: deltaX, yPixels: deltaY)
    }

    private func emitMaskGeometryConsoleLog(trigger: String, mask: UVLipAlphaMask?) {
        guard let mask else {
            print("PIPE-GEO trigger=\(trigger) mode=\(currentLipMode.rawValue) mask=nil")
            return
        }
        let bounds = Self.activeBounds(mask: mask, threshold: 24)
        let boundsText: String
        if let bounds {
            boundsText = String(
                format: "bounds=(%.1f,%.1f %.1fx%.1f)",
                bounds.minX,
                bounds.minY,
                bounds.width,
                bounds.height
            )
        } else {
            boundsText = "bounds=nil"
        }
        print(
            String(
                format: "PIPE-GEO trigger=%@ mode=%@ cov=%.2f%% max=%d %@",
                trigger,
                currentLipMode.rawValue,
                mask.nonzeroCoverage * 100,
                mask.maximumAlpha,
                boundsText
            )
        )
    }

    private func emitRenderConsoleLog(
        trigger: String,
        look: MakeupLook,
        intensity: Double,
        drawsLips: Bool,
        hasUVLipAlphaMask: Bool,
        lipAlphaMask: UVLipAlphaMask?,
        lipLineMask: UVLipAlphaMask?
    ) {
        let alphaCoverage = lipAlphaMask?.nonzeroCoverage ?? -1
        let alphaMax = lipAlphaMask.map { Int($0.maximumAlpha) } ?? -1
        let lineCoverage = lipLineMask?.nonzeroCoverage ?? -1
        let lineMax = lipLineMask.map { Int($0.maximumAlpha) } ?? -1
        print(
            String(
                format: "PIPE-RENDER t=%@ m=%@ look=%@ lipOnly=%d draw=%d dbg=%d i=%.2f hasUV=%d a=%.2f/%d l=%.2f/%d",
                trigger,
                currentLipMode.rawValue,
                look.name,
                currentLipOnlyTestMode ? 1 : 0,
                drawsLips ? 1 : 0,
                look.isLipLineDebug ? 1 : 0,
                intensity,
                hasUVLipAlphaMask ? 1 : 0,
                alphaCoverage * 100,
                alphaMax,
                lineCoverage * 100,
                lineMax
            )
        )
    }

    private static func renderedImageAlphaStats(_ image: UIImage) -> (coverage: CGFloat, maxAlpha: UInt8)? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        let sampleStride = max(min(width, height) / 128, 1)
        var active = 0
        var total = 0
        var maxAlpha: UInt8 = 0
        for y in Swift.stride(from: 0, to: height, by: sampleStride) {
            for x in Swift.stride(from: 0, to: width, by: sampleStride) {
                let alpha = pixels[(y * width + x) * 4 + 3]
                if alpha > 8 { active += 1 }
                maxAlpha = max(maxAlpha, alpha)
                total += 1
            }
        }
        guard total > 0 else { return nil }
        return (CGFloat(active) / CGFloat(total), maxAlpha)
    }

    private var latestScreenLipMask: ARKitLipMask?
    private var latestScreenLipMaskUpdatedAt: CFTimeInterval?
    private var latestScreenLipDisplayFrame: CGRect?
    private var screenLipMaskTracker = ScreenLipMaskTracker()
    private var latestFaceGeometry: ARFaceGeometry?
    private var latestFaceAnchor: ARFaceAnchor?
    private var latestLipOverlayFrame: CGRect?
    private var smoothedLipOverlayFrame: CGRect?
    private var latestLipParsingUpdatedAt: CFTimeInterval?
    private var latestLipParsingDurationMS: Double?
    private var latestLipParsingBreakdownText: String?
    private var lastPipelineConsoleLine: String?
    private var startupTraceID = 0
    private var startupTraceStartedAt: CFTimeInterval?
    private var didLogStartupFaceAnchor = false
    private var didLogStartupUVRequest = false
    private var didLogStartupHotStart = false
    private var didLogStartupTextureApplied = false
    private var hasRenderedRawUVUltraFastFirstFrame = false
    private var latestAutoMouthROIRatio: CGFloat?
    private var latestAutoUVMaskCoverage: CGFloat?
    private var latestAutoUVMaskMaximumAlpha: Int?
    private var latestLipRenderDurationMS: Double?
    private var latestLipOverlayGeometry: SCNGeometry?
    private weak var lipOverlayNode: SCNNode?
    private var viewportSize: CGSize = .zero
    private var performanceWindowStart = CACurrentMediaTime()
    private var performanceWindowFrames = 0
    private var renderedFPS: Double = 0
    private var lastPerformancePublishAt: CFTimeInterval = 0
    private var lastMainThreadTimingLogAt: CFTimeInterval = 0
    private var isSDFScheduleDispatchPending = false
    private var isSDFWorkerBusy = false
    private var lastHandledPhotoCaptureRequestID = 0
    private let screenLipMaskFreshDuration: CFTimeInterval = 0.22
    private let screenLipMaskFreezeDuration: CFTimeInterval = 0.70

    private var activeLipRefinementModel: ARKitLipRefinementModel? {
        if currentLipMode == .lipSDFRefinement {
            return lipSDFRefinementModel
        }
        return currentLipMode.usesLargeLipModel ? lipRefinementLargeModel : lipRefinementSmallModel
    }

    private var activeUVLipLineModel: ARKitLipRefinementModel {
        if currentLipMode == .lipSDFRefinement {
            return lipSDFRefinementModel
        }
        return lipLineAlphaModel
    }

    private var activeUVLipStudentModel: ARKitUVLipStudentModel {
        currentLipMode == .autoFusion || currentLipMode.usesUVSegStudent ? uvLipSegStudentModel : uvLipAlphaStudentModel
    }

    private var shouldDisplayUVLipAlphaMask: Bool {
        currentLipMode.usesUVLipStudent
            || currentLipMode.usesUVProjectedLipRefinement
            || (currentLipMode == .autoFusion && activeAutoLipRenderPath.usesUVAlphaMask)
    }

    private var shouldDisplayScreenLipOverlay: Bool {
        currentLipMode.showsScreenModelOverlay || (currentLipMode == .autoFusion && activeAutoLipRenderPath.usesScreenOverlay)
    }

    private var shouldAcceptUVLipStudentResult: Bool {
        currentLipMode.usesUVLipStudent
            || currentLipMode.usesUVProjectedLipRefinement
            || currentLipMode == .autoFusion
    }

    private var shouldAcceptScreenLipResult: Bool {
        currentLipMode.showsScreenModelOverlay || currentLipMode == .autoFusion
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        configureSceneView()
        configureMaterial()
        configureLipOverlayMaterial()
        configurePreciseLipMakeupLayer()
        configureLipParsingOverlay()
        uvSampleRecorder.onStatusChange = { [weak self] text in
            self?.onRecordingStatusChange?(text)
        }
        uvSampleRecorder.onRecordingActiveChange = { [weak self] active in
            self?.onRecordingActiveChange?(active)
        }
        warmUpUVLipLineAssistIfNeeded()
        warmUpMainLipModelsIfNeeded()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        sceneView.frame = bounds
        viewportSize = bounds.size
    }

    func startIfPossible() {
        guard !hasStarted else { return }
        guard ARFaceTrackingConfiguration.isSupported else {
            onStatusChange?(.unsupported)
            return
        }

        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        hasStarted = true
        if usesForcedRawUVPipeline && currentLipMode.usesUVSegStudent {
            resetTryOnStartupTrace(reason: "startIfPossible")
            logTryOnStartupEvent(
                stage: "session-run",
                details: "uvReady=\(activeUVLipStudentModel.isReady ? 1 : 0) viewport=\(Int(viewportSize.width))x\(Int(viewportSize.height))"
            )
        }
        onStatusChange?(.ready)
    }

    func pause() {
        sceneView.session.pause()
        hasStarted = false
        currentRecordARKitUVSamples = false
        uvSampleRecorder.stop()
    }

    func capturePhotoIfNeeded(requestID: Int) {
        guard requestID > 0, requestID != lastHandledPhotoCaptureRequestID else { return }
        lastHandledPhotoCaptureRequestID = requestID
        captureCurrentARFrame()
    }

    private func captureCurrentARFrame() {
        guard sceneView.bounds.width > 1, sceneView.bounds.height > 1 else {
            publishPhotoStatus("相机画面还没准备好")
            return
        }

        publishPhotoStatus("正在保存照片...")
        let image = makeCompositedPhotoSnapshot()
        UIImageWriteToSavedPhotosAlbum(
            image,
            self,
            #selector(photoSaveFinished(_:didFinishSavingWithError:contextInfo:)),
            nil
        )
    }

    private func makeCompositedPhotoSnapshot() -> UIImage {
        let sceneSnapshot = sceneView.snapshot()
        guard
            currentMakeupVisible,
            shouldDisplayScreenLipOverlay,
            latestScreenLipMask != nil
        else {
            return sceneSnapshot
        }

        let snapshotBounds = CGRect(origin: .zero, size: bounds.size)
        let format = UIGraphicsImageRendererFormat()
        format.scale = window?.screen.scale ?? UIScreen.main.scale
        format.opaque = true
        return UIGraphicsImageRenderer(size: bounds.size, format: format).image { context in
            UIColor.black.setFill()
            context.fill(snapshotBounds)
            sceneSnapshot.draw(in: snapshotBounds)
            renderPhotoLipMakeupOverlay()
        }
    }

    private func renderPhotoLipMakeupOverlay() {
        guard
            currentMakeupVisible,
            shouldDisplayScreenLipOverlay,
            let look = currentLook,
            let lipMask = latestScreenLipMask,
            let lipstickImage = lipMask.tintedImage(color: boostedLipUIColor(from: look))
        else {
            return
        }

        let targetFrame = latestScreenLipDisplayFrame ?? latestLipOverlayFrame ?? smoothedLipOverlayFrame ?? lipMask.frame
        UIImage(cgImage: lipstickImage).draw(in: targetFrame, blendMode: .normal, alpha: CGFloat(currentLipMode.modelOverlayOpacity))
    }

    @objc private func photoSaveFinished(
        _ image: UIImage,
        didFinishSavingWithError error: Error?,
        contextInfo: UnsafeRawPointer
    ) {
        if let error {
            publishPhotoStatus("照片保存失败：\(error.localizedDescription)")
        } else {
            publishPhotoStatus("已保存到相册")
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.publishPhotoStatus("")
        }
    }

    private func publishPhotoStatus(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onPhotoCaptureStatusChange?(text)
        }
    }

    func updateStyle(
        look: MakeupLook,
        intensity: Double,
        makeupVisible: Bool,
        lipMode: LipMakeupMode,
        showFaceMesh: Bool,
        lipOnlyTestMode: Bool,
        recordARKitUVSamples: Bool,
        forceRawUVOnly: Bool
    ) {
        let previousLipMode = currentLipMode
        let previousForceRawUVOnly = currentForceRawUVOnly
        let lookChanged = currentLook != look
        let intensityChanged = abs(currentIntensity - intensity) > 0.01
        let lipModeChanged = currentLipMode != lipMode
        let rawUVModeChanged = previousForceRawUVOnly != forceRawUVOnly
        let lipLineDebugRouteChanged = (currentLook?.lipLineDebugSource ?? .none) != look.lipLineDebugSource
        let changed = lookChanged
            || intensityChanged
            || currentMakeupVisible != makeupVisible
            || currentLipMode != lipMode
            || currentShowFaceMesh != showFaceMesh
            || currentLipOnlyTestMode != lipOnlyTestMode
            || rawUVModeChanged
            || currentRecordARKitUVSamples != recordARKitUVSamples

        currentLook = look
        currentIntensity = intensity
        currentMakeupVisible = makeupVisible
        currentLipMode = lipMode
        currentForceRawUVOnly = forceRawUVOnly
        currentShowFaceMesh = showFaceMesh
        currentLipOnlyTestMode = lipOnlyTestMode
        if currentRecordARKitUVSamples != recordARKitUVSamples {
            currentRecordARKitUVSamples = recordARKitUVSamples
            if recordARKitUVSamples {
                uvSampleRecorder.start()
            } else {
                uvSampleRecorder.stop()
            }
        }
        sceneView.debugOptions = showFaceMesh ? [.showWireframe] : []
        preciseLipMakeupLayer.opacity = lipMode.modelOverlayOpacity
        preciseLipMakeupLayer.isHidden = !(makeupVisible && shouldDisplayScreenLipOverlay && preciseLipMakeupLayer.contents != nil)
        lipParsingOverlayLayer.isHidden = !(makeupVisible && showFaceMesh && lipParsingOverlayLayer.contents != nil)

        let styleUpdateStartedAt = CACurrentMediaTime()
        guard changed else { return }
        if usesForcedRawUVPipeline, lipMode.usesUVSegStudent, latestUVLipAlphaMask == nil {
            resetTryOnStartupTrace(reason: "updateStyle changed")
            logTryOnStartupEvent(
                stage: "style-change",
                details: "lookChanged=\(lookChanged ? 1 : 0) intensityChanged=\(intensityChanged ? 1 : 0) lipModeChanged=\(lipModeChanged ? 1 : 0) rawChanged=\(rawUVModeChanged ? 1 : 0)"
            )
        }
        if lipModeChanged || rawUVModeChanged {
            // Cancel any in-flight parsing from the previous mode so the newly
            // selected route can schedule immediately. Older async callbacks are
            // ignored via generation mismatch once this advances.
            lipParsingGeneration += 1
            isLipParsingInFlight = false
            lastLipParsingStartedAt = 0
            let shouldCarryLipSDFMask = lipMode == .lipSDFRefinement && !usesForcedRawUVPipeline
            let carriedUVLipMask = shouldCarryLipSDFMask ? latestUVLipAlphaMask : nil
            let carriedUVLipLineMask = shouldCarryLipSDFMask ? latestUVLipLineRenderMask : nil
            latestFaceParsingMask = nil
            latestFaceUVMask = nil
            latestUVLipAlphaMask = nil
            latestUVLipSmoothingSourceMask = nil
            latestUVLipColorSupportMask = nil
            latestUVLipLineRenderMask = nil
            latestUVLipLineDebugMask = nil
            latestUVLipLineFusedDebugMask = nil
            latestUVLipLineOnlyDebugMask = nil
            latestUVLipLineColorDebugMask = nil
            latestCurveAssistMask = nil
            curveRejectStreak = 0
            curveBackoffFramesRemaining = 0
            uvLipMaskLock.reset()
            uvLipLineFusedMaskLock.reset()
            uvLipLineOnlyMaskLock.reset()
            uvLipLineColorMaskLock.reset()
            didResetUVLipMaskForCurveOuterV2 = false
            latestScreenLipMask = nil
            latestScreenLipMaskUpdatedAt = nil
            latestScreenLipDisplayFrame = nil
            screenLipMaskTracker.reset()
            activeAutoLipRenderPath = lipMode == .autoFusion ? .uv4c : .none
            latestLipParsingDurationMS = nil
            latestLipParsingBreakdownText = nil
            latestLipRenderDurationMS = nil
            latestLipOverlayGeometry = nil
            lipOverlayNode?.geometry = nil
            lipOverlayNode?.isHidden = true
            smoothedLipOverlayFrame = nil
            clearPreciseLipMakeupLayer()
            preciseLipMakeupLayer.isHidden = true
            if lipMode == .lipSDFRefinement {
                latestUVLipAlphaMask = carriedUVLipMask
                latestUVLipSmoothingSourceMask = carriedUVLipMask
                latestUVLipLineRenderMask = nil
            } else if usesForcedRawUVPipeline, lipMode.usesUVSegStudent {
                primeRawUVDebugHotStartIfPossible()
            }
            lastPipelineConsoleLine = nil
            emitPipelineConsoleLog(trigger: "mode-change", summary: "from=\(previousLipMode.rawValue) to=\(lipMode.rawValue)")
        }
        if lipLineDebugRouteChanged {
            latestUVLipLineDebugMask = cachedLipLineDebugMask(for: look.lipLineDebugSource)
        }
        if usesForcedRawUVPipeline && currentLipMode.usesUVSegStudent {
            textureRenderGeneration += 1
            if latestUVLipAlphaMask == nil {
                material.diffuse.contents = UIColor.clear
                primeRawUVDebugHotStartIfPossible()
            }
        }
        needsTextureRefresh = true
        refreshScreenLipMakeupLayer()
        refreshLatestLipParsingImage()
        updateLipOverlayVisibility()
        let canSwapRawUVShadeImmediately = (lookChanged || intensityChanged)
            && usesForcedRawUVPipeline
            && currentLipOnlyTestMode
            && latestUVLipAlphaMask != nil
        if canSwapRawUVShadeImmediately {
            renderFastRawUVShadePreviewIfPossible()
            if lookChanged {
                needsTextureRefresh = true
                updateMaterialTexture()
            } else {
                needsTextureRefresh = false
            }
            return
        }
        let prefersImmediateTextureRefresh = (lookChanged || intensityChanged)
            && !currentLipMode.usesUVLipStudent
            && !currentLipMode.usesUVProjectedLipRefinement
            && !(currentLipMode == .autoFusion && activeAutoLipRenderPath.usesUVAlphaMask)
        updateMaterialTexture(forceSynchronous: prefersImmediateTextureRefresh)
        let styleUpdateDurationMS = (CACurrentMediaTime() - styleUpdateStartedAt) * 1000
        emitMainThreadTimingLog(
            stage: "updateStyle",
            totalMS: styleUpdateDurationMS,
            thresholdMS: 12,
            details: "modeChanged=\(lipModeChanged ? 1 : 0) lookChanged=\(lookChanged ? 1 : 0) intensityChanged=\(intensityChanged ? 1 : 0) rawChanged=\(rawUVModeChanged ? 1 : 0)"
        )
    }

    private func cachedLipLineDebugMask(for source: LipLineDebugSource) -> UVLipAlphaMask? {
        switch source {
        case .none:
            nil
        case .fused:
            latestUVLipLineFusedDebugMask
        case .uvOnly:
            latestUVLipLineOnlyDebugMask
        case .colorOnly:
            latestUVLipLineColorDebugMask
        }
    }

    private func warmUpUVLipLineAssistIfNeeded() {
        guard enableUVLipLineAssist, !uvLipLineAssistWarmupStarted else { return }

        uvLipLineAssistWarmupStarted = true
        auxiliaryModelWarmupQueue.async { [weak self] in
            guard let self else { return }

            let isReady = self.lipLineAlphaModel.isReady
            DispatchQueue.main.async { [weak self] in
                self?.uvLipLineAssistReady = isReady
            }
        }
    }

    private func warmUpMainLipModelsIfNeeded() {
        guard !mainLipModelsWarmupStarted else { return }

        mainLipModelsWarmupStarted = true
        startupModelWarmupQueue.async { [weak self] in
            guard let self else { return }

            // Raw-UV try-on only needs the UV-4C segmentation student for the first visible result.
            _ = self.uvLipSegStudentModel.isReady

            self.auxiliaryModelWarmupQueue.async { [weak self] in
                guard let self else { return }

                _ = self.uvLipAlphaStudentModel.isReady
                _ = self.lipRefinementSmallModel.isReady
                _ = self.lipSDFRefinementModel.isReady
                _ = self.lipCurveModel.isReady
            }
        }
    }

    private func configureSceneView() {
        sceneView.translatesAutoresizingMaskIntoConstraints = false
        sceneView.backgroundColor = .black
        sceneView.automaticallyUpdatesLighting = false
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.scene = SCNScene()
        addSubview(sceneView)
        NSLayoutConstraint.activate([
            sceneView.leadingAnchor.constraint(equalTo: leadingAnchor),
            sceneView.trailingAnchor.constraint(equalTo: trailingAnchor),
            sceneView.topAnchor.constraint(equalTo: topAnchor),
            sceneView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    private func configureLipParsingOverlay() {
        lipParsingOverlayLayer.opacity = 0.34
        lipParsingOverlayLayer.isHidden = true
        lipParsingOverlayLayer.contentsGravity = .resize
        lipParsingOverlayLayer.minificationFilter = .linear
        lipParsingOverlayLayer.magnificationFilter = .linear
        lipParsingOverlayLayer.actions = [
            "bounds": NSNull(),
            "position": NSNull(),
            "contents": NSNull(),
            "opacity": NSNull(),
            "hidden": NSNull()
        ]
        layer.addSublayer(lipParsingOverlayLayer)
    }

    private func configurePreciseLipMakeupLayer() {
        preciseLipMakeupLayer.opacity = 0.94
        preciseLipMakeupLayer.isHidden = true
        preciseLipMakeupLayer.contentsScale = UIScreen.main.scale
        preciseLipMakeupLayer.contentsGravity = .resize
        preciseLipMakeupLayer.minificationFilter = .linear
        preciseLipMakeupLayer.magnificationFilter = .linear
        preciseLipMakeupLayer.actions = [
            "bounds": NSNull(),
            "position": NSNull(),
            "contents": NSNull(),
            "opacity": NSNull(),
            "hidden": NSNull()
        ]
        preciseLipMaskLayer.contentsGravity = .resize
        preciseLipMaskLayer.minificationFilter = .linear
        preciseLipMaskLayer.magnificationFilter = .linear
        preciseLipMaskLayer.actions = [
            "bounds": NSNull(),
            "position": NSNull(),
            "contents": NSNull()
        ]
        layer.addSublayer(preciseLipMakeupLayer)
    }

    private func configureMaterial() {
        material.lightingModel = .constant
        material.blendMode = .alpha
        material.transparencyMode = .aOne
        material.transparency = 1
        material.isDoubleSided = true
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false
        material.diffuse.mipFilter = .linear
        material.diffuse.minificationFilter = .linear
        material.diffuse.magnificationFilter = .linear
    }

    private func configureLipOverlayMaterial() {
        lipOverlayMaterial.lightingModel = .constant
        lipOverlayMaterial.blendMode = .alpha
        lipOverlayMaterial.transparencyMode = .aOne
        lipOverlayMaterial.transparency = 1
        lipOverlayMaterial.isDoubleSided = true
        lipOverlayMaterial.writesToDepthBuffer = false
        lipOverlayMaterial.readsFromDepthBuffer = false
        lipOverlayMaterial.diffuse.contents = UIColor.clear
        lipOverlayMaterial.emission.contents = UIColor.clear
        lipOverlayMaterial.diffuse.mipFilter = .linear
        lipOverlayMaterial.diffuse.minificationFilter = .linear
        lipOverlayMaterial.diffuse.magnificationFilter = .linear
        lipOverlayMaterial.shaderModifiers = [
            .fragment: """
            #pragma transparent
            _output.color = _geometry.color;
            """
        ]
    }

    private func updateMaterialTexture(forceSynchronous: Bool = false) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in
                guard let self, self.needsTextureRefresh else { return }
                self.updateMaterialTexture(forceSynchronous: forceSynchronous)
            }
            return
        }

        let textureUpdateStartedAt = CACurrentMediaTime()
        guard let look = currentLook else { return }
        let baseIntensity = currentLipMode == .lipRefinementLarge3DGPU ? currentIntensity * 0.46 : currentIntensity
        let meshIntensity = currentMakeupVisible ? baseIntensity : 0
        let hasModelLipMesh = currentLipMode.usesGPULipOverlay && latestLipOverlayGeometry != nil
        let autoUsesUVLipMask = currentLipMode == .autoFusion && activeAutoLipRenderPath.usesUVAlphaMask
        let hasUVLipAlphaMask = shouldDisplayUVLipAlphaMask && latestUVLipAlphaMask != nil
        let usesStartup3DLipFallback = currentLipMode.usesStartup3DLipFallback(
            hasUVLipAlphaMask: hasUVLipAlphaMask,
            isLipLineDebug: look.isLipLineDebug
        )
        let uvFaceParsingMask = currentLipMode == .faceParsing || (currentShowFaceMesh && !shouldDisplayUVLipAlphaMask)
        let drawsBaseLips = (currentLipMode.usesUVLipStudent || currentLipMode.usesUVProjectedLipRefinement)
            ? (hasUVLipAlphaMask || usesStartup3DLipFallback)
            : currentLipMode.uses3DLipBase
        let drawsAutoUVLips = autoUsesUVLipMask && hasUVLipAlphaMask
        let drawsLips = (drawsBaseLips || drawsAutoUVLips) && !hasModelLipMesh
        let shouldRenderTextureOffMain = currentLipMode.usesUVLipStudent
            || currentLipMode.usesUVProjectedLipRefinement
            || autoUsesUVLipMask
        let preserveClosedMouthCrease = latestMouthOpenAmount < 0.12
        let lipAlphaMask = shouldDisplayUVLipAlphaMask ? latestUVLipAlphaMask : nil
        let lipLineMask = look.isLipLineDebug ? latestUVLipLineDebugMask : latestUVLipLineRenderMask
        let forceRawUVDebugPath = usesForcedRawUVPipeline
            && (currentLipMode.usesUVSegStudent || currentLipMode == .autoFusion)
        // When switching into SDF, a blank sync render can land before the
        // first carried/hot-start mask arrives. Keep the previous texture until
        // we have a usable UV mask instead of flashing to an empty face.
        if currentLipMode.usesUVProjectedLipRefinement, lipAlphaMask == nil {
            needsTextureRefresh = true
            let durationMS = (CACurrentMediaTime() - textureUpdateStartedAt) * 1000
            emitMainThreadTimingLog(
                stage: "updateTexture",
                totalMS: durationMS,
                thresholdMS: 8,
                details: "earlyReturn=noMask async=\(shouldRenderTextureOffMain ? 1 : 0)"
            )
            return
        }
        if forceRawUVDebugPath && lipAlphaMask == nil {
            textureRenderGeneration += 1
            material.diffuse.contents = UIColor.clear
            needsTextureRefresh = false
            let durationMS = (CACurrentMediaTime() - textureUpdateStartedAt) * 1000
            emitMainThreadTimingLog(
                stage: "updateTexture",
                totalMS: durationMS,
                thresholdMS: 8,
                details: "earlyReturn=rawClear async=0"
            )
            return
        }
        emitRenderConsoleLog(
            trigger: shouldRenderTextureOffMain && !forceSynchronous ? "updateTexture-async" : "updateTexture-sync",
            look: look,
            intensity: meshIntensity,
            drawsLips: drawsLips,
            hasUVLipAlphaMask: hasUVLipAlphaMask,
            lipAlphaMask: lipAlphaMask,
            lipLineMask: lipLineMask
        )
        if
            forceRawUVDebugPath,
            currentLipOnlyTestMode,
            lipAlphaMask != nil
        {
            let renderStartedAt = CACurrentMediaTime()
            renderFastRawUVShadePreviewIfPossible()
            latestLipRenderDurationMS = (CACurrentMediaTime() - renderStartedAt) * 1000
            needsTextureRefresh = false
            isTextureRenderInFlight = false
            if !didLogStartupTextureApplied {
                didLogStartupTextureApplied = true
                logTryOnStartupEvent(
                    stage: "texture-applied",
                    details: "hasMask=1 cov=\(String(format: "%.2f", (lipAlphaMask?.nonzeroCoverage ?? 0) * 100)) max=\(lipAlphaMask?.maximumAlpha ?? 0) preview=1"
                )
            }
            return
        }
        if shouldRenderTextureOffMain && !forceSynchronous {
            // Coalesce rapid refreshes so an in-flight async render can actually land.
            if isTextureRenderInFlight {
                needsTextureRefresh = true
                return
            }
            textureRenderGeneration += 1
            let generation = textureRenderGeneration
            let lookSnapshot = look
            let intensitySnapshot = meshIntensity
            let faceParsingMask = uvFaceParsingMask ? latestFaceUVMask : nil
            let drawsFaceMakeup = !currentLipOnlyTestMode
            let showsDebugGuides = currentShowFaceMesh
            needsTextureRefresh = false
            isTextureRenderInFlight = true
            let renderStartedAt = CACurrentMediaTime()

            textureRenderQueue.async { [weak self] in
                let image = UVMakeupTexture.render(
                    look: lookSnapshot,
                    intensity: intensitySnapshot,
                    faceGeometry: nil,
                    faceParsingMask: faceParsingMask,
                    lipAlphaMask: lipAlphaMask,
                    lipLineMask: lipLineMask,
                    drawsLips: drawsLips,
                    drawsFaceMakeup: drawsFaceMakeup,
                    showsDebugGuides: showsDebugGuides,
                    preserveClosedMouthCrease: preserveClosedMouthCrease,
                    prefersTightLipMask: forceRawUVDebugPath
                )
                if let stats = Self.renderedImageAlphaStats(image) {
                    print(
                        String(
                            format: "PIPE-IMG t=async m=%@ cov=%.2f%% max=%d",
                            self?.currentLipMode.rawValue ?? "unknown",
                            stats.coverage * 100,
                            stats.maxAlpha
                        )
                    )
                } else {
                    print("PIPE-IMG t=async m=\(self?.currentLipMode.rawValue ?? "unknown") nil")
                }

                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.latestLipRenderDurationMS = (CACurrentMediaTime() - renderStartedAt) * 1000
                    self.isTextureRenderInFlight = false
                    guard self.textureRenderGeneration == generation else {
                        if self.needsTextureRefresh {
                            self.updateMaterialTexture()
                        }
                        return
                    }
                    self.material.diffuse.contents = image
                    if forceRawUVDebugPath, !self.didLogStartupTextureApplied {
                        self.didLogStartupTextureApplied = true
                        self.logTryOnStartupEvent(
                            stage: "texture-applied",
                            details: "hasMask=\(lipAlphaMask != nil ? 1 : 0) cov=\(String(format: "%.2f", (lipAlphaMask?.nonzeroCoverage ?? 0) * 100)) max=\(lipAlphaMask?.maximumAlpha ?? 0) async=1"
                        )
                    }
                    if self.needsTextureRefresh {
                        self.updateMaterialTexture()
                    }
                }
            }
            let durationMS = (CACurrentMediaTime() - textureUpdateStartedAt) * 1000
            emitMainThreadTimingLog(
                stage: "updateTexture",
                totalMS: durationMS,
                thresholdMS: 8,
                details: "path=async queued=\(isTextureRenderInFlight ? 1 : 0) alpha=\(lipAlphaMask != nil ? 1 : 0)"
            )
            return
        }

        // A style tap should feel instant. Invalidate any queued texture work so
        // an older debug-green render cannot land after the newly selected shade.
        textureRenderGeneration += 1
        let faceGeometryForTexture = shouldRenderTextureOffMain ? nil : latestFaceGeometry
        material.diffuse.contents = UVMakeupTexture.render(
            look: look,
            intensity: meshIntensity,
            faceGeometry: faceGeometryForTexture,
            faceParsingMask: uvFaceParsingMask ? latestFaceUVMask : nil,
            lipAlphaMask: lipAlphaMask,
            lipLineMask: lipLineMask,
            drawsLips: drawsLips,
            drawsFaceMakeup: !currentLipOnlyTestMode,
            showsDebugGuides: currentShowFaceMesh,
            preserveClosedMouthCrease: preserveClosedMouthCrease,
            prefersTightLipMask: forceRawUVDebugPath
        )
        if forceRawUVDebugPath, !didLogStartupTextureApplied {
            didLogStartupTextureApplied = true
            logTryOnStartupEvent(
                stage: "texture-applied",
                details: "hasMask=\(lipAlphaMask != nil ? 1 : 0) cov=\(String(format: "%.2f", (lipAlphaMask?.nonzeroCoverage ?? 0) * 100)) max=\(lipAlphaMask?.maximumAlpha ?? 0)"
            )
        }
        if let image = material.diffuse.contents as? UIImage,
           let stats = Self.renderedImageAlphaStats(image) {
            print(
                String(
                    format: "PIPE-IMG t=sync m=%@ cov=%.2f%% max=%d",
                    currentLipMode.rawValue,
                    stats.coverage * 100,
                    stats.maxAlpha
                )
            )
        } else {
            print("PIPE-IMG t=sync m=\(currentLipMode.rawValue) nil")
        }
        needsTextureRefresh = false
        let durationMS = (CACurrentMediaTime() - textureUpdateStartedAt) * 1000
        emitMainThreadTimingLog(
            stage: "updateTexture",
            totalMS: durationMS,
            thresholdMS: 8,
            details: "path=sync alpha=\(lipAlphaMask != nil ? 1 : 0) line=\(lipLineMask != nil ? 1 : 0)"
        )
    }

    private func renderFastRawUVShadePreviewIfPossible() {
        guard
            let look = currentLook,
            usesForcedRawUVPipeline,
            currentLipOnlyTestMode
        else {
            return
        }

        let prefersUltraFastFirstFrame = !hasRenderedRawUVUltraFastFirstFrame
        let previewMask = latestUVLipSmoothingSourceMask ?? latestUVLipAlphaMask
        guard let lipAlphaMask = previewMask else { return }

        textureRenderGeneration += 1
        material.diffuse.contents = UVMakeupTexture.renderFastLipOnlyPreview(
            look: look,
            intensity: currentMakeupVisible ? currentIntensity : 0,
            lipAlphaMask: lipAlphaMask,
            preserveClosedMouthCrease: latestMouthOpenAmount < 0.12
        )
        if prefersUltraFastFirstFrame {
            hasRenderedRawUVUltraFastFirstFrame = true
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard
            anchor is ARFaceAnchor,
            let device = sceneView.device,
            let geometry = ARSCNFaceGeometry(device: device, fillMesh: true)
        else {
            return nil
        }

        DispatchQueue.main.async { [weak self] in
            self?.resetLipTrackingStateForFaceLoss()
        }

        geometry.firstMaterial = material
        let node = SCNNode(geometry: geometry)

        let overlayNode = SCNNode()
        overlayNode.name = "gpu-lip-overlay"
        overlayNode.renderingOrder = 20
        overlayNode.isHidden = true
        node.addChildNode(overlayNode)
        lipOverlayNode = overlayNode
        return node
    }

    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        guard anchor is ARFaceAnchor else { return }

        DispatchQueue.main.async { [weak self] in
            self?.lipOverlayNode = nil
            self?.resetLipTrackingStateForFaceLoss()
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard
            let faceAnchor = anchor as? ARFaceAnchor,
            let geometry = node.geometry as? ARSCNFaceGeometry
        else {
            return
        }

        let didUpdateStartedAt = CACurrentMediaTime()
        geometry.update(from: faceAnchor.geometry)
        let geometryUpdateMS = (CACurrentMediaTime() - didUpdateStartedAt) * 1000
        latestFaceGeometry = faceAnchor.geometry
        latestFaceAnchor = faceAnchor
        latestMouthOpenAmount = mouthOpenAmount(for: faceAnchor)
        if usesForcedRawUVPipeline, currentLipMode.usesUVSegStudent, !didLogStartupFaceAnchor {
            didLogStartupFaceAnchor = true
            logTryOnStartupEvent(
                stage: "face-anchor",
                details: "uvReady=\(activeUVLipStudentModel.isReady ? 1 : 0) inFlight=\(isLipParsingInFlight ? 1 : 0) hot=\(isRawUVHotStartInFlight ? 1 : 0)"
            )
        }
        if
            usesForcedRawUVPipeline,
            currentLipMode.usesUVSegStudent,
            currentMakeupVisible,
            latestUVLipAlphaMask == nil,
            !isLipParsingInFlight
        {
            primeRawUVDebugHotStartIfPossible()
        }
        updateFrameCount += 1
        let currentFrame = sceneView.session.currentFrame
        if currentLipMode == .autoFusion, let currentFrame {
            updateAutoLipRenderPath(frame: currentFrame, faceAnchor: faceAnchor)
        }
        let postFrameSetupMS = (CACurrentMediaTime() - didUpdateStartedAt) * 1000
        if needsTextureRefresh {
            DispatchQueue.main.async { [weak self] in
                guard let self, self.needsTextureRefresh else { return }
                self.updateMaterialTexture()
            }
        }
        let shouldUpdateParsingOverlayFrame = currentShowFaceMesh
            || currentLipMode.showsScreenModelOverlay
            || (currentLipMode == .autoFusion && activeAutoLipRenderPath.usesScreenOverlay)
        if
            shouldUpdateParsingOverlayFrame,
            let currentFrame,
            let activeROI = activeParsingROI(frame: currentFrame, faceAnchor: faceAnchor)
        {
            updateLipParsingOverlayFrame(to: activeROI)
        }
        let overlayUpdateMS = (CACurrentMediaTime() - didUpdateStartedAt) * 1000
        if currentLipMode == .lipSDFRefinement {
            if !isSDFScheduleDispatchPending {
                isSDFScheduleDispatchPending = true
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.isSDFScheduleDispatchPending = false
                    self.scheduleLipParsingIfNeeded(
                        frame: self.sceneView.session.currentFrame,
                        faceAnchor: self.latestFaceAnchor ?? faceAnchor
                    )
                }
            }
        } else {
            scheduleLipParsingIfNeeded(frame: currentFrame, faceAnchor: faceAnchor)
        }
        let parsingScheduleMS = (CACurrentMediaTime() - didUpdateStartedAt) * 1000
        recordUVTrainingSampleIfNeeded(frame: currentFrame, faceAnchor: faceAnchor)

        recordTrackingFrame()
        let didUpdateTotalMS = (CACurrentMediaTime() - didUpdateStartedAt) * 1000
        emitMainThreadTimingLog(
            stage: "didUpdate",
            totalMS: didUpdateTotalMS,
            thresholdMS: 12,
            details: String(
                format: "geom=%.1f post=%.1f overlay=%.1f sched=%.1f tex=%d inFlight=%d",
                geometryUpdateMS,
                postFrameSetupMS,
                overlayUpdateMS,
                parsingScheduleMS,
                needsTextureRefresh ? 1 : 0,
                isLipParsingInFlight ? 1 : 0
            )
        )
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { [weak self] in
            self?.onStatusChange?(.failed(error.localizedDescription))
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in
            self?.onStatusChange?(.interrupted)
        }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in
            self?.hasStarted = false
            self?.resetLipTrackingStateForFaceLoss()
            self?.startIfPossible()
        }
    }

    @discardableResult
    private func updateAutoLipRenderPath(
        frame: ARFrame,
        faceAnchor: ARFaceAnchor,
        now: CFTimeInterval = CACurrentMediaTime()
    ) -> AutoLipRenderPath {
        guard currentLipMode == .autoFusion else { return activeAutoLipRenderPath }

        let mouthROI = projectedMouthROI(frame: frame, faceAnchor: faceAnchor)
        let mouthROIRatio = mouthROI.map {
            max($0.width / max(viewportSize.width, 1), $0.height / max(viewportSize.height, 1))
        }
        latestAutoMouthROIRatio = mouthROIRatio
        latestAutoUVMaskCoverage = latestUVLipAlphaMask?.nonzeroCoverage
        latestAutoUVMaskMaximumAlpha = latestUVLipAlphaMask.map { Int($0.maximumAlpha) }
        let screenMaskAge = latestScreenLipMaskUpdatedAt.map { now - $0 }
        let hasFreshScreenMask = latestScreenLipMask != nil
            && (screenMaskAge.map { $0 <= screenLipMaskFreshDuration } ?? false)
        let hasRecentScreenMask = latestScreenLipMask != nil
            && (screenMaskAge.map { $0 <= screenLipMaskFreezeDuration } ?? false)

        let nextPath = AutoLipRenderDecider.decide(
            mouthROIRatio: mouthROIRatio,
            uvMaskCoverage: latestUVLipAlphaMask?.nonzeroCoverage,
            uvMaskMaximumAlpha: latestUVLipAlphaMask.map { Int($0.maximumAlpha) },
            hasFreshScreenMask: hasFreshScreenMask,
            hasRecentScreenMask: hasRecentScreenMask,
            uvModelReady: uvLipSegStudentModel.isReady,
            screenModelReady: lipRefinementSmallModel.isReady
        )

        guard nextPath != activeAutoLipRenderPath else { return nextPath }

        activeAutoLipRenderPath = nextPath
        applyAutoLipRenderPath(nextPath)
        return nextPath
    }

    private func applyAutoLipRenderPath(_ path: AutoLipRenderPath) {
        guard currentLipMode == .autoFusion else { return }

        switch path {
        case .uv4c:
            clearPreciseLipMakeupLayer()
            updateLipOverlayGeometry(nil)
        case .screen2D, .freeze:
            updateLipOverlayGeometry(nil)
            if let latestScreenLipMask {
                updateScreenLipMakeupLayer(with: latestScreenLipMask)
            } else {
                clearPreciseLipMakeupLayer()
            }
        case .none:
            clearPreciseLipMakeupLayer()
            updateLipOverlayGeometry(nil)
        }

        needsTextureRefresh = true
    }

    private func autoLipDebugText() -> String {
        guard currentLipMode == .autoFusion else { return "" }

        let ratioText = latestAutoMouthROIRatio.map {
            String(format: " r%.2f", $0)
        } ?? " r--"
        let uvText: String
        if let coverage = latestAutoUVMaskCoverage, let maximumAlpha = latestAutoUVMaskMaximumAlpha {
            uvText = String(format: " uv%.1f%%/%d", coverage * 100, maximumAlpha)
        } else {
            uvText = " uv--"
        }

        return ratioText + uvText
    }

    private func recordTrackingFrame() {
        let now = CACurrentMediaTime()
        if
            currentLipMode == .lipSDFRefinement,
            isLipParsingInFlight,
            lastLipParsingStartedAt > 0,
            now - lastLipParsingStartedAt > 1.2
        {
            lipParsingGeneration += 1
            isLipParsingInFlight = false
            lastLipParsingStartedAt = 0
            latestLipParsingBreakdownText = "sdf watchdog reset"
            emitPipelineConsoleLog(trigger: "sdf-watchdog", summary: "timeout-reset")
        }
        performanceWindowFrames += 1
        let elapsed = now - performanceWindowStart
        guard elapsed >= 0.50 else { return }

        renderedFPS = Double(performanceWindowFrames) / elapsed
        performanceWindowFrames = 0
        performanceWindowStart = now

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.onStatusChange?(.tracking)
            guard
                (self.currentShowFaceMesh || self.currentLipMode.usesModel),
                now - self.lastPerformancePublishAt > 1.0
            else {
                return
            }
            self.lastPerformancePublishAt = now
            if self.currentLipMode == .autoFusion {
                let uvReady = self.uvLipSegStudentModel.isReady
                let screenReady = self.lipRefinementSmallModel.isReady
                if !uvReady && !screenReady {
                    self.onPerformanceChange?(String(format: "ARKit FPS %.0f  Auto models missing", self.renderedFPS))
                    return
                }
                if self.activeAutoLipRenderPath == .uv4c, !uvReady {
                    self.onPerformanceChange?(String(format: "ARKit FPS %.0f  Auto UV missing", self.renderedFPS))
                    return
                }
                if self.activeAutoLipRenderPath.usesScreenOverlay, !screenReady {
                    self.onPerformanceChange?(String(format: "ARKit FPS %.0f  Auto 2D missing", self.renderedFPS))
                    return
                }
            }
            if self.currentLipMode.usesLipROIModel, self.activeLipRefinementModel?.isReady == false {
                self.onPerformanceChange?(String(format: "ARKit FPS %.0f  Lip model missing", self.renderedFPS))
                return
            }
            if self.currentLipMode.usesUVLipStudent, !self.activeUVLipStudentModel.isReady {
                self.onPerformanceChange?(String(format: "ARKit FPS %.0f  UV model missing", self.renderedFPS))
                return
            }
            if let latestLipParsingDurationMS = self.latestLipParsingDurationMS {
                let mlLabel = self.currentLipMode == .autoFusion
                    ? "Auto \(self.activeAutoLipRenderPath.displayName)"
                    : (self.currentLipMode.performanceLabel ?? (self.currentShowFaceMesh ? "Face ML" : "ML"))
                let autoDebug = self.autoLipDebugText()
                let breakdown = self.latestLipParsingBreakdownText.map { "  " + $0 } ?? ""
                let renderDetail = self.latestLipRenderDurationMS.map {
                    String(format: self.shouldDisplayScreenLipOverlay || self.currentLipMode.usesGPULipOverlay ? " gpu %.0f" : " img %.0f", $0)
                } ?? ""
                self.onPerformanceChange?(
                    String(format: "ARKit FPS %.0f  %@%@ %.0fms%@%@", self.renderedFPS, mlLabel, autoDebug, latestLipParsingDurationMS, breakdown, renderDetail)
                )
            } else if self.currentLipMode == .lipSDFRefinement {
                let warmState: String
                if let carriedMask = self.latestUVLipAlphaMask {
                    warmState = String(
                        format: "Lip SDF holding %.2f%%/%d",
                        carriedMask.nonzeroCoverage * 100,
                        carriedMask.maximumAlpha
                    )
                } else if self.isLipParsingInFlight || self.isSDFHotStartInFlight || self.isSDFWorkerBusy {
                    warmState = "Lip SDF warming up"
                } else {
                    warmState = "Lip SDF waiting"
                }
                self.onPerformanceChange?(String(format: "ARKit FPS %.0f  %@", self.renderedFPS, warmState))
            } else {
                self.onPerformanceChange?(String(format: "ARKit FPS %.0f", self.renderedFPS))
            }
        }
    }

    private func recordUVTrainingSampleIfNeeded(frame: ARFrame?, faceAnchor: ARFaceAnchor) {
        guard currentRecordARKitUVSamples, let frame else { return }

        uvSampleRecorder.record(
            frame: frame,
            faceAnchor: faceAnchor,
            viewportSize: viewportSize,
            faceROI: projectedFaceROI(frame: frame, faceAnchor: faceAnchor),
            lipROI: projectedMouthROI(frame: frame, faceAnchor: faceAnchor)
        )
    }

    private func scheduleLipParsingIfNeeded(
        frame: ARFrame?,
        faceAnchor: ARFaceAnchor
    ) {
        guard
            currentMakeupVisible,
            !isLipParsingInFlight,
            currentLipMode.usesModel || currentShowFaceMesh,
            let frame
        else {
            return
        }

        let now = CACurrentMediaTime()

        if currentLipMode == .autoFusion {
            scheduleAutoFusionLipParsing(frame: frame, faceAnchor: faceAnchor)
            return
        }
        if currentLipMode == .lipSDFRefinement {
            let sdfInterval = latestLipParsingDurationMS.map {
                min(max(($0 / 1000.0) + 0.05, 0.12), 0.24)
            } ?? (latestUVLipAlphaMask == nil ? 0.12 : 0.18)
            if lastLipParsingStartedAt <= 0 {
                lastLipParsingStartedAt = now - sdfInterval - 0.001
            }
            guard !isSDFWorkerBusy else { return }
            guard now - lastLipParsingStartedAt > sdfInterval else { return }
            scheduleLipSDFFusionParsing(frame: frame, faceAnchor: faceAnchor, startedAt: now)
            return
        }

        let uvStudentModel = activeUVLipStudentModel
        let shouldRunUVLipStudent = currentLipMode.usesUVLipStudent && uvStudentModel.isReady
        let shouldRunLipRefinement = !shouldRunUVLipStudent && currentLipMode.usesLipROIModel && activeLipRefinementModel?.isReady == true
        let shouldRunFaceParsing = !shouldRunUVLipStudent && !shouldRunLipRefinement
            && (currentLipMode == .faceParsing || currentShowFaceMesh) && lipParsingModel.isReady
        guard shouldRunUVLipStudent || shouldRunLipRefinement || shouldRunFaceParsing else {
            if currentLipMode.isLipRefinement {
                preciseLipMakeupLayer.isHidden = true
            }
            return
        }

        let lipRefinementInterval = latestLipParsingDurationMS.map { min(max(($0 / 1000.0) + 0.020, 0.040), 0.10) } ?? 0.05
        let uvStudentInterval = latestLipParsingDurationMS.map { min(max(($0 / 1000.0) + 0.06, 0.12), 0.30) } ?? 0.12
        let needsFirstUVLipMask = shouldRunUVLipStudent && latestUVLipAlphaMask == nil
        let minimumInterval = needsFirstUVLipMask
            ? 0
            : (currentShowFaceMesh ? 0.45 : (shouldRunUVLipStudent ? uvStudentInterval : (shouldRunLipRefinement ? lipRefinementInterval : 0.80)))
        if lastLipParsingStartedAt <= 0 {
            lastLipParsingStartedAt = now - minimumInterval - 0.001
        }
        guard needsFirstUVLipMask || now - lastLipParsingStartedAt > minimumInterval else { return }

        if shouldRunUVLipStudent {
            if latestUVLipAlphaMask != nil, !currentLipMode.usesUVSegStudent, !isFacePoseStableForUVCalibration(faceAnchor) {
                return
            }
            scheduleUVLipStudentParsing(frame: frame, faceAnchor: faceAnchor, startedAt: now, model: uvStudentModel)
        } else if shouldRunLipRefinement {
            scheduleLipRefinementParsing(frame: frame, faceAnchor: faceAnchor, startedAt: now)
        } else {
            scheduleFaceParsing(frame: frame, faceAnchor: faceAnchor, startedAt: now)
        }
    }

    private func scheduleAutoFusionLipParsing(
        frame: ARFrame,
        faceAnchor: ARFaceAnchor
    ) {
        let now = CACurrentMediaTime()
        let activePath = updateAutoLipRenderPath(frame: frame, faceAnchor: faceAnchor, now: now)
        let shouldRunUVLipStudent = activePath.usesUVAlphaMask && uvLipSegStudentModel.isReady
        let shouldRunLipRefinement = activePath.usesScreenOverlay && lipRefinementSmallModel.isReady
        guard shouldRunUVLipStudent || shouldRunLipRefinement else {
            if activePath == .none {
                hideStaleLipParsingOverlay()
            }
            return
        }

        let lipRefinementInterval = latestLipParsingDurationMS.map { min(max(($0 / 1000.0) + 0.020, 0.040), 0.10) } ?? 0.05
        let uvStudentInterval = latestLipParsingDurationMS.map { min(max(($0 / 1000.0) + 0.06, 0.12), 0.30) } ?? 0.12
        let needsFirstUVLipMask = shouldRunUVLipStudent && latestUVLipAlphaMask == nil
        let minimumInterval = needsFirstUVLipMask ? 0 : (shouldRunUVLipStudent ? uvStudentInterval : lipRefinementInterval)
        if lastLipParsingStartedAt <= 0 {
            lastLipParsingStartedAt = now - minimumInterval - 0.001
        }
        guard needsFirstUVLipMask || now - lastLipParsingStartedAt > minimumInterval else { return }

        if shouldRunUVLipStudent {
            if latestUVLipAlphaMask != nil, activePath != .uv4c, !isFacePoseStableForUVCalibration(faceAnchor) {
                return
            }
            scheduleUVLipStudentParsing(frame: frame, faceAnchor: faceAnchor, startedAt: now, model: uvLipSegStudentModel)
        } else if shouldRunLipRefinement {
            scheduleLipRefinementParsing(frame: frame, faceAnchor: faceAnchor, startedAt: now)
        }
    }

    private func scheduleUVLipStudentParsing(
        frame: ARFrame,
        faceAnchor: ARFaceAnchor,
        startedAt now: CFTimeInterval,
        model uvLipStudentModel: ARKitUVLipStudentModel
    ) {
        guard
            let request = uvLipStudentModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                arFrame: frame,
                faceAnchor: faceAnchor,
                viewportSize: viewportSize
            )
        else {
            return
        }
        if usesForcedRawUVPipeline, currentLipMode.usesUVSegStudent, !didLogStartupUVRequest {
            didLogStartupUVRequest = true
            logTryOnStartupEvent(
                stage: "uv-request",
                details: "viewport=\(Int(viewportSize.width))x\(Int(viewportSize.height)) mouthOpen=\(String(format: "%.3f", latestMouthOpenAmount))"
            )
        }

        let currentLookSnapshot = currentLook
        let isLipLineDebugLook = currentLookSnapshot?.isLipLineDebug == true
        let mouthROI = projectedMouthROI(frame: frame, faceAnchor: faceAnchor)
        let mouthOpenAmount = latestMouthOpenAmount
        let preserveClosedMouthCrease = mouthOpenAmount < 0.12
        let forceRawUVDebugPath = usesForcedRawUVPipeline
            && (currentLipMode.usesUVSegStudent || currentLipMode == .autoFusion)
        let shouldBypassUVLock = !forceRawUVDebugPath && mouthOpenAmount > 0.28
        if curveBackoffFramesRemaining > 0 {
            curveBackoffFramesRemaining -= 1
        }
        curveAssistFrameCounter &+= 1
        let curveAssistStride = mouthOpenAmount > 0.10 ? 2 : 3
        let shouldRunCurveAssist = !forceRawUVDebugPath
            && currentLipMode.usesUVSegStudent
            && lipCurveModel.isReady
            && curveBackoffFramesRemaining == 0
            && (latestCurveAssistMask == nil
                || latestUVLipAlphaMask == nil
                || curveAssistFrameCounter.isMultiple(of: curveAssistStride))
        let cachedCurveAssistMask = latestCurveAssistMask
        let canRunLipLineAssist = !forceRawUVDebugPath
            && enableUVLipLineAssist
            && uvLipLineAssistReady
            && currentLipMode.usesUVSegStudent
            && latestUVLipAlphaMask != nil
            && isLipLineDebugLook
        let sharedProjectionMap = (shouldRunCurveAssist || canRunLipLineAssist) && mouthROI != nil
            ? faceUVProjectionMap(frame: frame, faceAnchor: faceAnchor)
            : nil
        let curveProjectionMap: ARKitFaceUVProjectionMap?
        let curveRequest: ARKitLipParsingRequest?
        let curveModel: ARKitLipCurveModel?
        // UV-4C reuses the curve model as a boundary refinement assist instead of exposing a separate mode.
        if shouldRunCurveAssist {
            curveProjectionMap = sharedProjectionMap
            if let mouthROI {
                curveRequest = lipCurveModel.makeRequest(
                    pixelBuffer: frame.capturedImage,
                    roi: mouthROI,
                    arFrame: frame,
                    viewportSize: viewportSize
                )
            } else {
                curveRequest = nil
            }
            curveModel = lipCurveModel
        } else {
            curveProjectionMap = nil
            curveRequest = nil
            curveModel = nil
        }
        let lipLineProjectionMap: ARKitFaceUVProjectionMap?
        let lipLineRequest: ARKitLipParsingRequest?
        let lipLineModel: ARKitLipRefinementModel?
        if canRunLipLineAssist {
            let activeLineModel = activeUVLipLineModel
            if activeLineModel.isReady {
                lipLineProjectionMap = sharedProjectionMap
                if let mouthROI {
                    lipLineRequest = activeLineModel.makeRequest(
                        pixelBuffer: frame.capturedImage,
                        roi: mouthROI,
                        arFrame: frame,
                        viewportSize: viewportSize
                    )
                } else {
                    lipLineRequest = nil
                }
                lipLineModel = activeLineModel
            } else {
                lipLineProjectionMap = nil
                lipLineRequest = nil
                lipLineModel = nil
            }
        } else {
            lipLineProjectionMap = nil
            lipLineRequest = nil
            lipLineModel = nil
        }

        let shouldRunLipColorSupport = !forceRawUVDebugPath && currentLipMode.usesUVSegStudent && isLipLineDebugLook
        if currentLipMode.usesUVSegStudent, curveModel != nil, !didResetUVLipMaskForCurveOuterV2 {
            didResetUVLipMaskForCurveOuterV2 = true
            latestUVLipAlphaMask = nil
            latestUVLipSmoothingSourceMask = nil
            latestUVLipColorSupportMask = nil
            latestUVLipLineRenderMask = nil
            latestUVLipLineDebugMask = nil
            latestUVLipLineFusedDebugMask = nil
            latestUVLipLineOnlyDebugMask = nil
            latestUVLipLineColorDebugMask = nil
            uvLipMaskLock.reset()
            uvLipLineFusedMaskLock.reset()
            uvLipLineOnlyMaskLock.reset()
            uvLipLineColorMaskLock.reset()
        }
        let previousMask = latestUVLipAlphaMask
        let smoothingResponse: CGFloat = mouthOpenAmount > 0.10 ? 0.94 : 0.84
        let edgeRefinement: UVLipEdgeRefinement = currentLipMode.usesUVSegStudent ? .uvSegProduct : .product
        let parsingGeneration = lipParsingGeneration
        isLipParsingInFlight = true
        lastLipParsingStartedAt = now
        lipParsingQueue.async { [weak self, uvLipStudentModel, request, curveModel, curveRequest, curveProjectionMap, lipLineModel, lipLineRequest, lipLineProjectionMap, shouldRunLipColorSupport, previousMask, smoothingResponse, edgeRefinement, parsingGeneration, preserveClosedMouthCrease, shouldBypassUVLock, mouthOpenAmount, curveAssistStride, shouldRunCurveAssist, cachedCurveAssistMask, forceRawUVDebugPath] in
            let start = CACurrentMediaTime()
            let timedMask = uvLipStudentModel.timedLipAlphaMask(request: request)
            var lipColorSupportMask: UVLipAlphaMask?
            var lipColorStatus = "color off"
            if shouldRunLipColorSupport, let candidateMask = timedMask.mask {
                let timedColor = uvLipStudentModel.timedLipColorSupportMask(request: request, candidate: candidateMask)
                lipColorSupportMask = timedColor.mask
                lipColorStatus = String(format: "%@ %.0f", timedColor.info, timedColor.ms)
            }
            var lipLineStatus = "line off"
            if let lipLineModel, let lipLineRequest, let lipLineProjectionMap {
                let lineStart = CACurrentMediaTime()
                let timedLineMask = lipLineModel.timedMask(request: lipLineRequest)
                let lineMS = (CACurrentMediaTime() - lineStart) * 1000
                if let lineMask = timedLineMask.mask, let uvLineMask = lipLineProjectionMap.uvLipAlphaMask(from: lineMask) {
                    lipLineStatus = String(format: "line ok %.0f %.2f%%/%d", lineMS, uvLineMask.nonzeroCoverage * 100, uvLineMask.maximumAlpha)
                } else if timedLineMask.mask == nil {
                    lipLineStatus = String(format: "line mask nil %.0f", lineMS)
                } else {
                    lipLineStatus = String(format: "line uv nil %.0f", lineMS)
                }
            }
            var curveMask: UVLipAlphaMask?
            var curveMaskIsFresh = false
            var curveStatus = shouldRunCurveAssist ? "curveOuterV2 pending" : "curveOuterV2 skip/\(curveAssistStride)"
            if let curveModel, let curveRequest, let curveProjectionMap {
                let curveStart = CACurrentMediaTime()
                let timedCurve = curveModel.timedMask(request: curveRequest, projectionMap: curveProjectionMap)
                let curveMS = (CACurrentMediaTime() - curveStart) * 1000
                if let uvCurveMask = timedCurve.uvMask
                    ?? timedCurve.mask.flatMap({ curveProjectionMap.uvLipAlphaMask(from: $0) }) {
                    curveMaskIsFresh = true
                    let stabilizedCurveMask = uvCurveMask.temporallySmoothed(
                        previous: cachedCurveAssistMask,
                        response: mouthOpenAmount > 0.10 ? 0.82 : 0.46
                    )
                    curveMask = stabilizedCurveMask
                    curveStatus = String(
                        format: "curveOuterV2 ok %.0f %.2f%%/%d %@",
                        curveMS,
                        stabilizedCurveMask.nonzeroCoverage * 100,
                        stabilizedCurveMask.maximumAlpha,
                        timedCurve.info
                    )
                } else if timedCurve.mask == nil {
                    curveMask = cachedCurveAssistMask
                    curveStatus = String(
                        format: cachedCurveAssistMask == nil
                            ? "curveOuterV2 mask nil %.0f %@"
                            : "curveOuterV2 hold nil %.0f %@",
                        curveMS,
                        timedCurve.info
                    )
                } else {
                    curveMask = cachedCurveAssistMask
                    curveStatus = String(
                        format: cachedCurveAssistMask == nil
                            ? "curveOuterV2 uv nil %.0f %@"
                            : "curveOuterV2 hold uvNil %.0f %@",
                        curveMS,
                        timedCurve.info
                    )
                }
            } else if let cachedCurveAssistMask {
                curveMask = cachedCurveAssistMask
                curveStatus = "curveOuterV2 hold/\(curveAssistStride)"
            }
            let processedMask: UVLipAlphaMask?
            let cachedCurveForNextFrame: UVLipAlphaMask?
            let curvePrimaryApplied: Bool
            let curvePipelineSource: String
            var pipelineStatus: String
            if let lipAlphaMask = timedMask.mask {
                let smoothedMask = lipAlphaMask.temporallySmoothed(previous: previousMask, response: smoothingResponse)
                let silhouetteMask = preserveClosedMouthCrease
                    ? smoothedMask.preservedClosedMouthCrease()
                    : smoothedMask
                let featheredMask = self?.uvLipMaskFeather.feathered(silhouetteMask, refinement: edgeRefinement) ?? silhouetteMask
                let baseMask = preserveClosedMouthCrease
                    ? featheredMask.preservedClosedMouthCrease()
                    : featheredMask
                let effectiveCurveMask: UVLipAlphaMask?
                if let curveMask,
                   (curveMaskIsFresh || Self.curveAssistMaskIsReusable(baseMask: baseMask, curveMask: curveMask)) {
                    effectiveCurveMask = curveMask
                } else {
                    effectiveCurveMask = nil
                    if curveMask != nil {
                        curveStatus += " stale"
                    }
                }
                if forceRawUVDebugPath {
                    cachedCurveForNextFrame = nil
                    curvePrimaryApplied = false
                    curvePipelineSource = "raw-uv"
                    processedMask = baseMask
                    pipelineStatus = " curvePrimaryRawUV"
                    curveStatus = "curveOuterV2 forced-off"
                } else if let effectiveCurveMask {
                    let curveApplied = Self.curvePrimaryLipMask(
                        baseMask: baseMask,
                        curveMask: effectiveCurveMask,
                        mouthOpenAmount: mouthOpenAmount
                    )
                    if let curveApplied {
                        cachedCurveForNextFrame = effectiveCurveMask
                        curvePrimaryApplied = true
                        curvePipelineSource = "curve-primary"
                        processedMask = curveApplied.mask
                        pipelineStatus = String(format: " curvePrimary %.2f %@", curveApplied.overlap, curveApplied.reason)
                    } else {
                        cachedCurveForNextFrame = cachedCurveAssistMask
                        curvePrimaryApplied = false
                        curvePipelineSource = "uv"
                        processedMask = baseMask
                        pipelineStatus = Self.curvePrimaryLipMaskRejectStatus(
                            baseMask: baseMask,
                            curveMask: effectiveCurveMask,
                            mouthOpenAmount: mouthOpenAmount
                        )
                    }
                } else {
                    curvePrimaryApplied = false
                    curvePipelineSource = "uv"
                    cachedCurveForNextFrame = cachedCurveAssistMask
                    processedMask = baseMask
                    pipelineStatus = " curvePrimaryNone"
                }
            } else {
                curvePrimaryApplied = false
                curvePipelineSource = "none"
                cachedCurveForNextFrame = nil
                processedMask = nil
                pipelineStatus = " curvePrimaryNoBase"
            }
            let modelMS = (CACurrentMediaTime() - start) * 1000

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard self.lipParsingGeneration == parsingGeneration else {
                    self.isLipParsingInFlight = false
                    self.lastLipParsingStartedAt = 0
                    return
                }
                self.isLipParsingInFlight = false
                self.latestLipParsingDurationMS = modelMS
                let maskStatus: String
                if let mask = timedMask.mask {
                    maskStatus = String(format: " mask %.2f%%/%d", mask.nonzeroCoverage * 100, mask.maximumAlpha)
                } else {
                    maskStatus = " mask nil"
                }
                self.latestLipParsingBreakdownText = String(
                    format: "prep %.0f pred %.0f dec %.0f raw %.0f post %.0f %@ %@%@ %@ %@ %@",
                    timedMask.inputMS,
                    timedMask.predictionMS,
                    timedMask.decodeMS,
                    timedMask.decodeRawMS,
                    timedMask.decodePostMS,
                    timedMask.outputDataType,
                    timedMask.decodeInfo,
                    maskStatus,
                    lipLineStatus,
                    curveStatus,
                    lipColorStatus
                )

                guard
                    self.currentMakeupVisible,
                    self.shouldAcceptUVLipStudentResult,
                    self.currentLipMode != .lipSDFRefinement,
                    let processedMask
                else {
                    self.emitPipelineConsoleLog(trigger: "uv-branch-empty")
                    if self.currentMakeupVisible, self.shouldAcceptUVLipStudentResult, self.latestUVLipAlphaMask != nil {
                        return
                    }
                    self.hideStaleLipParsingOverlay()
                    return
                }

                self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                self.latestUVLipColorSupportMask = lipColorSupportMask
                if self.currentLipMode.usesUVSegStudent && !forceRawUVDebugPath {
                    self.latestCurveAssistMask = cachedCurveForNextFrame
                    if curvePrimaryApplied {
                        self.curveRejectStreak = 0
                        self.curveBackoffFramesRemaining = 0
                    } else if curveStatus.contains("curveOuterV2 ok") || curveStatus.contains("curveOuterV2 hold") || curveStatus.contains("curveOuterV2 uv nil") || curveStatus.contains("curveOuterV2 mask nil") {
                        self.curveRejectStreak += 1
                        if self.curveRejectStreak >= 3 {
                            self.curveBackoffFramesRemaining = 18
                            self.curveRejectStreak = 0
                        }
                    }
                } else {
                    self.latestCurveAssistMask = nil
                    self.curveRejectStreak = 0
                    self.curveBackoffFramesRemaining = 0
                }
                var displayedMask: UVLipAlphaMask
                let lockStatus: String
                if self.currentLipMode.usesUVSegStudent {
                    if forceRawUVDebugPath {
                        if self.currentLipOnlyTestMode {
                            if mouthOpenAmount > 0.28 {
                                self.uvLipMaskLock.reset()
                                displayedMask = processedMask
                                self.latestUVLipSmoothingSourceMask = processedMask
                                lockStatus = String(format: " uvLock open%.2f", mouthOpenAmount)
                            } else {
                                let lockResult = self.uvLipMaskLock.resolve(candidate: processedMask)
                                displayedMask = self.stabilizeRawUVArea(
                                    reference: lockResult.mask,
                                    candidate: processedMask
                                )
                                self.latestUVLipSmoothingSourceMask = displayedMask
                                lockStatus = " uvLock area \(lockResult.decision.statusText)"
                            }
                        } else {
                            self.uvLipMaskLock.reset()
                            displayedMask = processedMask
                            self.latestUVLipSmoothingSourceMask = processedMask
                            lockStatus = " uvLock raw-direct"
                        }
                    } else if shouldBypassUVLock {
                        self.uvLipMaskLock.reset()
                        displayedMask = processedMask
                        self.latestUVLipSmoothingSourceMask = processedMask
                        lockStatus = String(format: " uvLock open%.2f", mouthOpenAmount)
                    } else if curvePrimaryApplied {
                        self.uvLipMaskLock.reset()
                        displayedMask = processedMask
                        self.latestUVLipSmoothingSourceMask = processedMask
                        lockStatus = " uvLock curve"
                    } else {
                        let lockResult = self.uvLipMaskLock.resolve(candidate: processedMask)
                        displayedMask = lockResult.mask
                        self.latestUVLipSmoothingSourceMask = displayedMask
                        lockStatus = " uvLock \(lockResult.decision.statusText)"
                    }
                } else {
                    displayedMask = processedMask
                    lockStatus = ""
                }
                if forceRawUVDebugPath, self.currentLipOnlyTestMode {
                    let tightenedMask = displayedMask.contracted(
                        threshold: 24,
                        radius: 1,
                        minimumNeighborRatio: 0.74
                    )
                    displayedMask = Self.preparedUVFillMask(
                        fillMask: tightenedMask,
                        previousMask: self.latestUVLipAlphaMask,
                        contourBase: tightenedMask,
                        preserveClosedMouthCrease: mouthOpenAmount < 0.12,
                        mouthOpenAmount: mouthOpenAmount
                    )
                }
                self.latestUVLipSmoothingSourceMask = displayedMask
                self.latestUVLipAlphaMask = displayedMask
                let finalSourceStatus: String
                if self.currentLipMode.usesUVSegStudent {
                    if forceRawUVDebugPath {
                        finalSourceStatus = " finalSrc raw-uv"
                    } else if shouldBypassUVLock {
                        finalSourceStatus = curvePrimaryApplied
                            ? " finalSrc \(curvePipelineSource)"
                            : " finalSrc uv-open"
                    } else {
                        finalSourceStatus = " finalSrc \(curvePrimaryApplied ? curvePipelineSource : "uv")"
                    }
                } else {
                    finalSourceStatus = " finalSrc other"
                }
                let colorLineStatus: String
                if self.currentLook?.isLipLineDebug == true {
                    let uvLipLineSelection = UVLipLineCandidateSelector.select(
                        displayedMask: displayedMask,
                        colorSupportMask: lipColorSupportMask,
                        source: .uvOnly
                    )
                    let uvLipLineLock: UVLipMaskLockResult
                    if shouldBypassUVLock {
                        self.uvLipLineOnlyMaskLock.reset()
                        uvLipLineLock = UVLipMaskLockResult(mask: uvLipLineSelection.mask, decision: .relocked)
                    } else {
                        uvLipLineLock = self.uvLipLineOnlyMaskLock.resolve(candidate: uvLipLineSelection.mask)
                    }
                    self.latestUVLipLineOnlyDebugMask = uvLipLineLock.mask

                    var activeLineSelection = uvLipLineSelection
                    var activeLineLock = uvLipLineLock
                    if let lipColorSupportMask {
                        let colorLipLineSelection = UVLipLineCandidateSelector.select(
                            displayedMask: displayedMask,
                            colorSupportMask: lipColorSupportMask,
                            source: .colorOnly
                        )
                        let colorLipLineLock: UVLipMaskLockResult
                        if shouldBypassUVLock {
                            self.uvLipLineColorMaskLock.reset()
                            colorLipLineLock = UVLipMaskLockResult(mask: colorLipLineSelection.mask, decision: .relocked)
                        } else {
                            colorLipLineLock = self.uvLipLineColorMaskLock.resolve(candidate: colorLipLineSelection.mask)
                        }
                        self.latestUVLipLineColorDebugMask = colorLipLineLock.mask

                        let fusedLipLineSelection = UVLipLineCandidateSelector.select(
                            displayedMask: displayedMask,
                            colorSupportMask: lipColorSupportMask,
                            source: .fused
                        )
                        let fusedLipLineLock: UVLipMaskLockResult
                        if shouldBypassUVLock {
                            self.uvLipLineFusedMaskLock.reset()
                            fusedLipLineLock = UVLipMaskLockResult(mask: fusedLipLineSelection.mask, decision: .relocked)
                        } else {
                            fusedLipLineLock = self.uvLipLineFusedMaskLock.resolve(candidate: fusedLipLineSelection.mask)
                        }
                        self.latestUVLipLineFusedDebugMask = fusedLipLineLock.mask

                        switch self.currentLook?.lipLineDebugSource ?? .none {
                        case .fused:
                            activeLineSelection = fusedLipLineSelection
                            activeLineLock = fusedLipLineLock
                        case .colorOnly:
                            activeLineSelection = colorLipLineSelection
                            activeLineLock = colorLipLineLock
                        case .none, .uvOnly:
                            break
                        }
                    }
                    self.latestUVLipLineDebugMask = self.cachedLipLineDebugMask(for: self.currentLook?.lipLineDebugSource ?? .none)
                    colorLineStatus = "\(activeLineSelection.status) \(activeLineLock.decision.statusText)"
                } else {
                    self.latestUVLipLineOnlyDebugMask = nil
                    self.latestUVLipLineColorDebugMask = nil
                    self.latestUVLipLineFusedDebugMask = nil
                    self.latestUVLipLineRenderMask = nil
                    self.latestUVLipLineDebugMask = nil
                    colorLineStatus = ""
                }
                let finalMaskStatus = String(
                    format: " final %.2f%%/%d%@%@%@%@",
                    displayedMask.nonzeroCoverage * 100,
                    displayedMask.maximumAlpha,
                    pipelineStatus,
                    lockStatus,
                    finalSourceStatus,
                    colorLineStatus
                )
                self.latestLipParsingBreakdownText = (self.latestLipParsingBreakdownText ?? "") + finalMaskStatus
                self.emitPipelineConsoleLog(trigger: "uv-branch-final")
                self.latestLipRenderDurationMS = nil
                self.updateLipOverlayGeometry(nil)
                self.needsTextureRefresh = true
                self.updateMaterialTexture()
                if self.shouldDisplayUVLipAlphaMask {
                    self.clearPreciseLipMakeupLayer()
                }
                self.lipParsingOverlayLayer.isHidden = true
            }
        }
    }

    private static func curvePrimaryLipMask(
        baseMask: UVLipAlphaMask,
        curveMask: UVLipAlphaMask,
        mouthOpenAmount: CGFloat
    ) -> (mask: UVLipAlphaMask, overlap: CGFloat, reason: String)? {
        guard
            baseMask.width == curveMask.width,
            baseMask.height == curveMask.height,
            baseMask.alpha.count == curveMask.alpha.count,
            baseMask.maximumAlpha > 16,
            curveMask.maximumAlpha > 16
        else {
            return nil
        }

        guard let normalizedCurve = normalizedCurveOuterMask(baseMask: baseMask, curveMask: curveMask) else {
            return nil
        }
        let candidateCurveMask = normalizedCurve.mask.productAlphaSmoothedForMakeup(edgeThreshold: 24)
        let coverageRatio = normalizedCurve.coverageRatio
        guard coverageRatio > 0.30, coverageRatio < 1.90 else { return nil }
        let anchorGate = UVLipLineGeometryGate.evaluate(
            reference: baseMask,
            candidate: candidateCurveMask,
            maximumCornerShiftRatio: 0.20,
            maximumCenterShiftRatio: 0.14,
            minimumSideBalanceRatio: 0.16,
            minimumWidthRatio: 0.58,
            maximumWidthRatio: 1.24,
            minimumHeightRatio: 0.52,
            maximumHeightRatio: 1.28
        )
        guard anchorGate.passed else { return nil }
        let lipRegionGuard = baseMask.expanded(threshold: 24, radius: 1, blurRadius: 2)
        let guardedCurveMask = candidateCurveMask.clipped(to: lipRegionGuard)

        let threshold: UInt8 = 24
        var baseCount = 0
        var curveCount = 0
        var overlapCount = 0
        for index in guardedCurveMask.alpha.indices {
            let baseActive = baseMask.alpha[index] > threshold
            let curveActive = guardedCurveMask.alpha[index] > threshold
            if baseActive { baseCount += 1 }
            if curveActive { curveCount += 1 }
            if baseActive && curveActive { overlapCount += 1 }
        }
        guard baseCount > 32, curveCount > 32 else { return nil }

        let overlapRatio = CGFloat(overlapCount) / CGFloat(max(min(baseCount, curveCount), 1))
        guard overlapRatio > 0.30 else {
            return nil
        }
        let coverageGate = UVLipCoverageGate.evaluate(
            reference: baseMask,
            candidate: guardedCurveMask,
            minimumCoverage: 0.38,
            maximumSpillRate: 0.52,
            minimumAreaRatio: 0.44,
            maximumAreaRatio: 1.42,
            minimumIOU: 0.28
        )
        guard coverageGate.passed else {
            return nil
        }

        var composedMask = guardedCurveMask
        let holeMask: UVLipAlphaMask?
        if mouthOpenAmount > 0.08 {
            holeMask = baseMask.enclosedHoleMask(
                threshold: 24,
                expansionRadius: mouthOpenAmount > 0.22 ? 1 : 0,
                blurRadius: 1
            )
        } else {
            holeMask = nil
        }
        if let holeMask {
            composedMask = composedMask.subtracting(holeMask)
        }

        return (
            composedMask,
            overlapRatio,
            String(
                format: "cov%.2f exp%d shrink%d wx%.2f hy%.2f a%.2f g%.2f%@",
                coverageRatio,
                normalizedCurve.expandPasses,
                normalizedCurve.shrinkPasses,
                normalizedCurve.widthScale ?? 1,
                normalizedCurve.heightScale ?? 1,
                anchorGate.heightRatio,
                coverageGate.areaRatio,
                holeMask == nil ? "" : " hole"
            )
        )
    }

    private static func curvePrimaryFallbackMask(
        baseMask: UVLipAlphaMask,
        curveMask: UVLipAlphaMask,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        var composedMask = curveMask.productAlphaSmoothedForMakeup(edgeThreshold: 24)
        if mouthOpenAmount > 0.08,
           let holeMask = baseMask.enclosedHoleMask(
               threshold: 24,
               expansionRadius: mouthOpenAmount > 0.22 ? 1 : 0,
               blurRadius: 1
           ) {
            composedMask = composedMask.subtracting(holeMask)
        }
        return composedMask
    }

    private static func curvePrimaryLipMaskRejectStatus(
        baseMask: UVLipAlphaMask,
        curveMask: UVLipAlphaMask,
        mouthOpenAmount _: CGFloat
    ) -> String {
        guard
            baseMask.width == curveMask.width,
            baseMask.height == curveMask.height,
            baseMask.alpha.count == curveMask.alpha.count
        else {
            return " curveOuterReject size"
        }

        guard baseMask.maximumAlpha > 16 else {
            return " curveOuterReject baseLow"
        }
        guard curveMask.maximumAlpha > 16 else {
            return " curveOuterReject curveLow"
        }

        let baseCoverage = max(baseMask.nonzeroCoverage, 0.0001)
        guard let normalizedCurve = normalizedCurveOuterMask(baseMask: baseMask, curveMask: curveMask) else {
            return " curveOuterReject constrain pre"
        }

        let candidateCurveMask = normalizedCurve.mask
        let curveCoverage = candidateCurveMask.nonzeroCoverage
        let coverageRatio = curveCoverage / baseCoverage
        if coverageRatio <= 0.30 || coverageRatio >= 1.90 {
            return String(format: " curvePrimaryReject cov%.2f b%.2f c%.2f e%d s%d wx%.2f hy%.2f", coverageRatio, baseCoverage * 100, curveCoverage * 100, normalizedCurve.expandPasses, normalizedCurve.shrinkPasses, normalizedCurve.widthScale ?? 1, normalizedCurve.heightScale ?? 1)
        }
        let anchorGate = UVLipLineGeometryGate.evaluate(
            reference: baseMask,
            candidate: candidateCurveMask,
            maximumCornerShiftRatio: 0.20,
            maximumCenterShiftRatio: 0.14,
            minimumSideBalanceRatio: 0.16,
            minimumWidthRatio: 0.58,
            maximumWidthRatio: 1.24,
            minimumHeightRatio: 0.52,
            maximumHeightRatio: 1.28
        )
        if !anchorGate.passed {
            return " curvePrimaryReject" + anchorGate.statusText
        }
        let lipRegionGuard = baseMask.expanded(threshold: 24, radius: 1, blurRadius: 2)
        let guardedCurveMask = candidateCurveMask.clipped(to: lipRegionGuard)

        let threshold: UInt8 = 24
        var baseCount = 0
        var curveCount = 0
        var overlapCount = 0
        for index in guardedCurveMask.alpha.indices {
            let baseActive = baseMask.alpha[index] > threshold
            let curveActive = guardedCurveMask.alpha[index] > threshold
            if baseActive { baseCount += 1 }
            if curveActive { curveCount += 1 }
            if baseActive && curveActive { overlapCount += 1 }
        }
        guard baseCount > 32 else {
            return String(format: " curveOuterReject baseCount%d", baseCount)
        }
        guard curveCount > 32 else {
            return String(format: " curveOuterReject curveCount%d", curveCount)
        }

        let overlapRatio = CGFloat(overlapCount) / CGFloat(max(min(baseCount, curveCount), 1))
        if overlapRatio <= 0.30 {
            return String(format: " curvePrimaryReject overlap%.2f cov%.2f e%d s%d wx%.2f hy%.2f", overlapRatio, coverageRatio, normalizedCurve.expandPasses, normalizedCurve.shrinkPasses, normalizedCurve.widthScale ?? 1, normalizedCurve.heightScale ?? 1)
        }
        let coverageGate = UVLipCoverageGate.evaluate(
            reference: baseMask,
            candidate: guardedCurveMask,
            minimumCoverage: 0.38,
            maximumSpillRate: 0.52,
            minimumAreaRatio: 0.44,
            maximumAreaRatio: 1.42,
            minimumIOU: 0.28
        )
        if !coverageGate.passed {
            return " curvePrimaryReject gate" + coverageGate.statusText + String(format: " wx%.2f hy%.2f", normalizedCurve.widthScale ?? 1, normalizedCurve.heightScale ?? 1)
        }

        return String(format: " curvePrimaryReject unknown overlap%.2f cov%.2f e%d s%d wx%.2f hy%.2f", overlapRatio, coverageRatio, normalizedCurve.expandPasses, normalizedCurve.shrinkPasses, normalizedCurve.widthScale ?? 1, normalizedCurve.heightScale ?? 1)
    }

    private static func curveAssistMaskIsReusable(
        baseMask: UVLipAlphaMask,
        curveMask: UVLipAlphaMask
    ) -> Bool {
        guard
            baseMask.width == curveMask.width,
            baseMask.height == curveMask.height,
            baseMask.alpha.count == curveMask.alpha.count,
            curveMask.maximumAlpha > 16
        else {
            return false
        }

        let anchorGate = UVLipLineGeometryGate.evaluate(
            reference: baseMask,
            candidate: curveMask,
            maximumCornerShiftRatio: 0.24,
            maximumCenterShiftRatio: 0.18,
            minimumSideBalanceRatio: 0.12,
            minimumWidthRatio: 0.48,
            maximumWidthRatio: 1.36,
            minimumHeightRatio: 0.40,
            maximumHeightRatio: 1.46
        )
        guard anchorGate.passed else { return false }

        let coverageGate = UVLipCoverageGate.evaluate(
            reference: baseMask,
            candidate: curveMask,
            minimumCoverage: 0.30,
            maximumSpillRate: 0.64,
            minimumAreaRatio: 0.36,
            maximumAreaRatio: 1.72,
            minimumIOU: 0.20
        )
        return coverageGate.passed
    }

    private static func normalizedCurveOuterMask(
        baseMask: UVLipAlphaMask,
        curveMask: UVLipAlphaMask
    ) -> (mask: UVLipAlphaMask, coverageRatio: CGFloat, expandPasses: Int, shrinkPasses: Int, widthScale: CGFloat?, heightScale: CGFloat?)? {
        guard
            baseMask.width == curveMask.width,
            baseMask.height == curveMask.height,
            baseMask.alpha.count == curveMask.alpha.count
        else {
            return nil
        }

        let baseCoverage = max(baseMask.nonzeroCoverage, 0.0001)
        var candidate = curveMask.productAlphaSmoothedForMakeup(edgeThreshold: 24)
        guard candidate.maximumAlpha > 16 else { return nil }
        var widthScaleApplied: CGFloat?
        var heightScaleApplied: CGFloat?
        if let baseBounds = activeBounds(mask: baseMask, threshold: 24),
           let candidateBounds = activeBounds(mask: candidate, threshold: 24) {
            let baseWidth = max(baseBounds.width, 1)
            let baseHeight = max(baseBounds.height, 1)
            let candidateWidth = max(candidateBounds.width, 1)
            let candidateHeight = max(candidateBounds.height, 1)
            let widthRatio = candidateWidth / baseWidth
            let heightRatio = candidateHeight / baseHeight
            let needsWidthScale = widthRatio > 1.14 || widthRatio < 0.86
            let needsHeightScale = heightRatio > 1.12 || heightRatio < 0.88
            if needsWidthScale || needsHeightScale {
                let targetWidth = min(max(baseWidth * 0.92, candidateWidth * 0.72), baseWidth * 1.08)
                let targetHeight = min(max(baseHeight * 0.90, candidateHeight * 0.68), baseHeight * 1.06)
                let scaleX = needsWidthScale ? (targetWidth / candidateWidth) : 1
                let scaleY = needsHeightScale ? (targetHeight / candidateHeight) : 1
                candidate = scaledMask(
                    candidate,
                    centerX: candidateBounds.midX,
                    centerY: candidateBounds.midY,
                    targetCenterX: baseBounds.midX,
                    targetCenterY: baseBounds.midY,
                    scaleX: scaleX,
                    scaleY: scaleY,
                    threshold: 24
                ).productAlphaSmoothedForMakeup(edgeThreshold: 24)
                widthScaleApplied = needsWidthScale ? scaleX : nil
                heightScaleApplied = needsHeightScale ? scaleY : nil
            }
        }
        let lipRegionGuard = baseMask.expanded(threshold: 24, radius: 1, blurRadius: 2)
        candidate = candidate.clipped(to: lipRegionGuard)
        var expandPasses = 0
        var shrinkPasses = 0
        var coverageRatio = candidate.nonzeroCoverage / baseCoverage
        while coverageRatio < 0.82, expandPasses < 2 {
            candidate = candidate.expanded(threshold: 24, radius: 1, blurRadius: 1)
            expandPasses += 1
            coverageRatio = candidate.nonzeroCoverage / baseCoverage
        }
        while coverageRatio > 1.04, shrinkPasses < 3 {
            candidate = candidate.contracted(threshold: 24, radius: 1, minimumNeighborRatio: 0.74)
            shrinkPasses += 1
            coverageRatio = candidate.nonzeroCoverage / baseCoverage
        }

        return (candidate, coverageRatio, expandPasses, shrinkPasses, widthScaleApplied, heightScaleApplied)
    }

    private static func activeBounds(mask: UVLipAlphaMask, threshold: UInt8) -> CGRect? {
        guard mask.width > 0, mask.height > 0, mask.alpha.count == mask.width * mask.height else { return nil }

        var minX = mask.width
        var maxX = -1
        var minY = mask.height
        var maxY = -1
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask.alpha[y * mask.width + x] > threshold {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }

        guard maxX >= minX, maxY >= minY else { return nil }
        return CGRect(
            x: CGFloat(minX),
            y: CGFloat(minY),
            width: CGFloat(maxX - minX + 1),
            height: CGFloat(maxY - minY + 1)
        )
    }

    private static func scaledMask(
        _ mask: UVLipAlphaMask,
        centerX: CGFloat,
        centerY: CGFloat,
        targetCenterX: CGFloat,
        targetCenterY: CGFloat,
        scaleX: CGFloat,
        scaleY: CGFloat,
        threshold: UInt8
    ) -> UVLipAlphaMask {
        guard
            mask.width > 1,
            mask.height > 1,
            mask.alpha.count == mask.width * mask.height,
            scaleX.isFinite,
            scaleY.isFinite,
            scaleX > 0.01,
            scaleY > 0.01
        else {
            return mask
        }

        var output = [UInt8](repeating: 0, count: mask.alpha.count)
        for y in 0..<mask.height {
            for x in 0..<mask.width {
                let alpha = mask.alpha[y * mask.width + x]
                guard alpha > threshold else { continue }
                let mappedX = targetCenterX + (CGFloat(x) - centerX) * scaleX
                let mappedY = targetCenterY + (CGFloat(y) - centerY) * scaleY
                let roundedX = Int(mappedX.rounded())
                let roundedY = Int(mappedY.rounded())
                guard roundedX >= 0, roundedX < mask.width, roundedY >= 0, roundedY < mask.height else { continue }
                let index = roundedY * mask.width + roundedX
                output[index] = max(output[index], alpha)
                if roundedX + 1 < mask.width {
                    output[roundedY * mask.width + roundedX + 1] = max(output[roundedY * mask.width + roundedX + 1], UInt8(max(Int(alpha) - 32, 0)))
                }
                if roundedX - 1 >= 0 {
                    output[roundedY * mask.width + roundedX - 1] = max(output[roundedY * mask.width + roundedX - 1], UInt8(max(Int(alpha) - 32, 0)))
                }
                if roundedY + 1 < mask.height {
                    output[(roundedY + 1) * mask.width + roundedX] = max(output[(roundedY + 1) * mask.width + roundedX], UInt8(max(Int(alpha) - 32, 0)))
                }
                if roundedY - 1 >= 0 {
                    output[(roundedY - 1) * mask.width + roundedX] = max(output[(roundedY - 1) * mask.width + roundedX], UInt8(max(Int(alpha) - 32, 0)))
                }
            }
        }

        return UVLipAlphaMask(width: mask.width, height: mask.height, alpha: output)
    }

    private func isFacePoseStableForUVCalibration(_ faceAnchor: ARFaceAnchor) -> Bool {
        facePoseNode.simdTransform = faceAnchor.transform
        let yaw = abs(CGFloat(facePoseNode.eulerAngles.y))
        let pitch = abs(CGFloat(facePoseNode.eulerAngles.x))
        let roll = abs(CGFloat(facePoseNode.eulerAngles.z))
        return yaw < 0.35 && pitch < 0.32 && roll < 0.42
    }

    private func mouthOpenAmount(for faceAnchor: ARFaceAnchor) -> CGFloat {
        func blendShape(_ location: ARFaceAnchor.BlendShapeLocation) -> CGFloat {
            guard let value = faceAnchor.blendShapes[location] else { return 0 }
            return CGFloat(truncating: value)
        }

        let jawOpen = blendShape(.jawOpen)
        let mouthFunnel = blendShape(.mouthFunnel)
        let mouthPucker = blendShape(.mouthPucker)
        return min(max(max(jawOpen, mouthFunnel * 0.55), mouthPucker * 0.35), 1)
    }

    private func scheduleFaceParsing(frame: ARFrame, faceAnchor: ARFaceAnchor, startedAt now: CFTimeInterval) {
        guard
            let faceROI = projectedFaceROI(frame: frame, faceAnchor: faceAnchor),
            let request = lipParsingModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                roi: faceROI,
                arFrame: frame,
                viewportSize: viewportSize
            )
        else {
            return
        }

        isLipParsingInFlight = true
        lastLipParsingStartedAt = now
        let projectionMap = currentShowFaceMesh ? faceUVProjectionMap(frame: frame, faceAnchor: faceAnchor) : nil
        let parsingGeneration = lipParsingGeneration
        lipParsingQueue.async { [weak self, lipParsingModel, request, projectionMap, parsingGeneration] in
            let start = CACurrentMediaTime()
            let mask = lipParsingModel.faceMask(request: request)
            let uvMask = mask.flatMap { projectionMap?.uvMask(from: $0) }
            let durationMS = (CACurrentMediaTime() - start) * 1000

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard self.lipParsingGeneration == parsingGeneration else {
                    self.isLipParsingInFlight = false
                    self.lastLipParsingStartedAt = 0
                    return
                }
                self.isLipParsingInFlight = false
                self.latestLipParsingDurationMS = durationMS
                self.latestLipParsingBreakdownText = nil

                guard
                    self.currentMakeupVisible,
                    let mask
                else {
                    self.hideStaleLipParsingOverlay()
                    return
                }

                self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                self.latestFaceParsingMask = mask
                if self.currentLipMode == .faceParsing, let lipMask = mask.lipstickMask() {
                    self.updateScreenLipMakeupLayer(with: lipMask)
                } else {
                    self.preciseLipMakeupLayer.isHidden = true
                }
                if let uvMask {
                    let smoothedUVMask = uvMask.temporallySmoothed(previous: self.latestFaceUVMask, response: 0.42)
                    self.latestFaceUVMask = smoothedUVMask
                    if self.currentLipMode == .faceParsing || self.currentShowFaceMesh {
                        self.needsTextureRefresh = true
                        self.updateMaterialTexture()
                    }
                }

                guard
                    self.currentShowFaceMesh,
                    let image = self.faceParsingCompositor.makeDebugImage(from: mask) ?? mask.debugImage()
                else {
                    self.lipParsingOverlayLayer.isHidden = true
                    return
                }

                self.lipParsingOverlayLayer.frame = self.smoothedLipOverlayFrame ?? self.latestLipOverlayFrame ?? mask.frame
                self.lipParsingOverlayLayer.contents = image
                self.lipParsingOverlayLayer.isHidden = false
            }
        }
    }

    private func scheduleLipRefinementParsing(frame: ARFrame, faceAnchor: ARFaceAnchor, startedAt now: CFTimeInterval) {
        guard let lipRefinementModel = activeLipRefinementModel else { return }
        let shouldProjectLipMaskToUV = currentLipMode.usesUVProjectedLipRefinement
        guard
            let mouthROI = projectedMouthROI(frame: frame, faceAnchor: faceAnchor),
            let request = lipRefinementModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                roi: mouthROI,
                arFrame: frame,
                viewportSize: viewportSize
            )
        else {
            return
        }

        isLipParsingInFlight = true
        lastLipParsingStartedAt = now
        let usesGPULipOverlay = currentLipMode.usesGPULipOverlay
        let projectionMap = (usesGPULipOverlay || shouldProjectLipMaskToUV)
            ? faceUVProjectionMap(frame: frame, faceAnchor: faceAnchor)
            : nil
        let projectionVertices = usesGPULipOverlay ? lipMeshProjectionVertices(frame: frame, faceAnchor: faceAnchor, maskFrame: mouthROI) : nil
        let projectionColor = usesGPULipOverlay ? currentLook.map { lipProjectionColor(for: $0) } : nil
        let parsingGeneration = lipParsingGeneration
        lipParsingQueue.async { [weak self, lipRefinementModel, request, usesGPULipOverlay, shouldProjectLipMaskToUV, projectionMap, projectionVertices, projectionColor, parsingGeneration] in
            let start = CACurrentMediaTime()
            let timedMask = lipRefinementModel.timedMask(request: request)
            let lipMask = timedMask.mask
            let modelMS = (CACurrentMediaTime() - start) * 1000
            let projectionStatus: String
            if !usesGPULipOverlay {
                projectionStatus = "uv off"
            } else if lipMask == nil {
                projectionStatus = "gpu no-mask"
            } else if projectionVertices == nil {
                projectionStatus = "gpu no-vtx"
            } else if projectionColor == nil {
                projectionStatus = "gpu no-color"
            } else {
                projectionStatus = "mesh ready"
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard self.lipParsingGeneration == parsingGeneration else {
                    self.isLipParsingInFlight = false
                    self.lastLipParsingStartedAt = 0
                    return
                }
                var overlayGeometry: SCNGeometry?
                var renderDurationMS: Double?
                var finalProjectionStatus = projectionStatus
                if
                    usesGPULipOverlay,
                    let lipMask,
                    let projectionVertices,
                    let projectionColor
                {
                    if let meshResult = self.lipMeshProjector.makeGeometry(
                        mask: lipMask,
                        vertices: projectionVertices,
                        lipColor: projectionColor
                    ) {
                        overlayGeometry = meshResult.geometry
                        renderDurationMS = meshResult.durationMS
                        finalProjectionStatus = "mesh ok"
                    } else {
                        finalProjectionStatus = "mesh miss"
                    }
                }

                self.isLipParsingInFlight = false
                self.latestLipParsingDurationMS = modelMS
                self.latestLipParsingBreakdownText = String(
                    format: usesGPULipOverlay
                        ? "prep %.0f pred %.0f dec %.0f raw %.0f post %.0f %@ %@"
                        : "prep %.0f pred %.0f dec %.0f raw %.0f post %.0f %@ %@",
                    timedMask.inputMS,
                    timedMask.predictionMS,
                    timedMask.decodeMS,
                    timedMask.decodeRawMS,
                    timedMask.decodePostMS,
                    timedMask.outputDataType,
                    finalProjectionStatus
                )

                guard
                    self.currentMakeupVisible,
                    self.shouldAcceptScreenLipResult
                else {
                    self.hideStaleLipParsingOverlay()
                    return
                }

                guard let lipMask else {
                    if self.currentLipMode == .autoFusion, self.hasRecentScreenLipMask() {
                        self.activeAutoLipRenderPath = .freeze
                        _ = self.updateTrackedScreenLipMakeupFrame(now: CACurrentMediaTime())
                        self.needsTextureRefresh = true
                        self.updateMaterialTexture()
                    } else {
                        self.clearPreciseLipMakeupLayer()
                    }
                    self.lipParsingOverlayLayer.isHidden = true
                    self.updateLipOverlayGeometry(nil)
                    return
                }

                if shouldProjectLipMaskToUV,
                   let projectionMap,
                   let projectedMask = projectionMap.uvLipAlphaMask(from: lipMask)
                {
                    self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                    self.latestUVLipColorSupportMask = nil
                    self.latestUVLipAlphaMask = projectedMask
                    self.latestUVLipSmoothingSourceMask = projectedMask
                    if self.currentLook?.isLipLineDebug == true {
                        self.latestUVLipLineDebugMask = self.cachedLipLineDebugMask(for: self.currentLook?.lipLineDebugSource ?? .none)
                    } else {
                        self.latestUVLipLineDebugMask = nil
                        self.latestUVLipLineOnlyDebugMask = nil
                        self.latestUVLipLineColorDebugMask = nil
                        self.latestUVLipLineFusedDebugMask = nil
                        self.latestUVLipLineRenderMask = nil
                    }
                    self.latestLipRenderDurationMS = nil
                    self.updateLipOverlayGeometry(nil)
                    self.needsTextureRefresh = true
                    self.updateMaterialTexture()
                    self.clearPreciseLipMakeupLayer()
                    self.lipParsingOverlayLayer.isHidden = true
                    return
                }

                self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                if self.shouldDisplayScreenLipOverlay {
                    self.updateScreenLipMakeupLayer(with: lipMask)
                } else {
                    self.latestLipRenderDurationMS = nil
                    self.clearPreciseLipMakeupLayer()
                }
                if self.currentLipMode.usesGPULipOverlay {
                    self.latestLipRenderDurationMS = renderDurationMS
                    self.updateLipOverlayGeometry(overlayGeometry)
                } else {
                    self.updateLipOverlayGeometry(nil)
                }
                self.lipParsingOverlayLayer.isHidden = true
            }
        }
    }

    private func scheduleLipSDFFusionParsing(
        frame: ARFrame,
        faceAnchor: ARFaceAnchor,
        startedAt now: CFTimeInterval
    ) {
        guard
            uvLipSegStudentModel.isReady,
            lipSDFRefinementModel.isReady,
            let mouthROI = projectedMouthROI(frame: frame, faceAnchor: faceAnchor),
            let fillRequest = uvLipSegStudentModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                arFrame: frame,
                faceAnchor: faceAnchor,
                viewportSize: viewportSize
            ),
            let sdfRequest = lipSDFRefinementModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                roi: mouthROI,
                arFrame: frame,
                viewportSize: viewportSize
            ),
            let projectionMap = faceUVProjectionMap(frame: frame, faceAnchor: faceAnchor)
        else {
            return
        }
        let previousMask = latestUVLipAlphaMask
        let previousLineMask = latestUVLipLineRenderMask
        let mouthOpenAmount = latestMouthOpenAmount
        let preserveClosedMouthCrease = mouthOpenAmount < 0.12
        let shouldBypassUVLock = mouthOpenAmount > 0.16
        let shouldRunStartupAssist = false
        if curveBackoffFramesRemaining > 0 {
            curveBackoffFramesRemaining -= 1
        }
        curveAssistFrameCounter &+= 1
        let curveAssistStride = mouthOpenAmount > 0.10 ? 2 : 3
        let shouldRunCurveAssist = false
        let cachedCurveAssistMask: UVLipAlphaMask? = nil
        let parsingGeneration = lipParsingGeneration
        isLipParsingInFlight = true
        isSDFWorkerBusy = true
        lastLipParsingStartedAt = now
        let templateRequest = shouldRunStartupAssist && lipRefinementSmallModel.isReady
            ? lipRefinementSmallModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                roi: mouthROI,
                arFrame: frame,
                viewportSize: viewportSize
            )
            : nil
        let curveRequest: ARKitLipParsingRequest? = if shouldRunCurveAssist {
            lipCurveModel.makeRequest(
                pixelBuffer: frame.capturedImage,
                roi: mouthROI,
                arFrame: frame,
                viewportSize: viewportSize
            )
        } else {
            nil
        }

        sdfLipParsingQueue.async { [weak self, uvLipSegStudentModel, lipSDFRefinementModel, lipRefinementSmallModel, fillRequest, sdfRequest, projectionMap, templateRequest, curveRequest, previousMask, previousLineMask, mouthOpenAmount, preserveClosedMouthCrease, shouldBypassUVLock, parsingGeneration, shouldRunCurveAssist, cachedCurveAssistMask, curveAssistStride] in
            guard let self else { return }
            let start = CACurrentMediaTime()
            let timedFill = uvLipSegStudentModel.timedLipAlphaMask(request: fillRequest)
            let timedSDF = lipSDFRefinementModel.timedMask(request: sdfRequest)
            let modelMS = (CACurrentMediaTime() - start) * 1000
            let sdfMask = timedSDF.mask.flatMap { projectionMap.uvLipAlphaMask(from: $0) }
            let templateFillMask: UVLipAlphaMask?
            let templateLineMask: UVLipAlphaMask?
            let templateStatus: String
            if let templateRequest {
                let timedTemplate = lipRefinementSmallModel.timedMask(request: templateRequest)
                if let templateMask = timedTemplate.mask,
                   let projectedTemplateMask = projectionMap.uvLipAlphaMask(from: templateMask) {
                    let preparedTemplateMask = Self.preparedTemplateWarmStartMask(
                        fillMask: projectedTemplateMask,
                        previousMask: previousMask,
                        preserveClosedMouthCrease: preserveClosedMouthCrease,
                        mouthOpenAmount: mouthOpenAmount
                    )
                    templateFillMask = preparedTemplateMask
                    templateLineMask = Self.templateWarmStartLineMask(
                        fillMask: preparedTemplateMask,
                        previousLineMask: previousLineMask,
                        mouthOpenAmount: mouthOpenAmount
                    )
                    templateStatus = String(format: "template %.2f%%/%d", preparedTemplateMask.nonzeroCoverage * 100, preparedTemplateMask.maximumAlpha)
                } else {
                    templateFillMask = nil
                    templateLineMask = nil
                    templateStatus = "template nil"
                }
            } else {
                templateFillMask = nil
                templateLineMask = nil
                templateStatus = "template off"
            }
            var curveMask: UVLipAlphaMask?
            var curveMaskIsFresh = false
            var curveStatus = shouldRunCurveAssist ? "curveOuterV2 pending" : "curveOuterV2 skip/\(curveAssistStride)"
            if let curveRequest {
                let curveStart = CACurrentMediaTime()
                let timedCurve = self.lipCurveModel.timedMask(request: curveRequest, projectionMap: projectionMap)
                let curveMS = (CACurrentMediaTime() - curveStart) * 1000
                if let uvCurveMask = timedCurve.uvMask
                    ?? timedCurve.mask.flatMap({ projectionMap.uvLipAlphaMask(from: $0) }) {
                    curveMaskIsFresh = true
                    let stabilizedCurveMask = uvCurveMask.temporallySmoothed(
                        previous: cachedCurveAssistMask,
                        response: mouthOpenAmount > 0.10 ? 0.82 : 0.46
                    )
                    curveMask = stabilizedCurveMask
                    curveStatus = String(
                        format: "curveOuterV2 ok %.0f %.2f%%/%d %@",
                        curveMS,
                        stabilizedCurveMask.nonzeroCoverage * 100,
                        stabilizedCurveMask.maximumAlpha,
                        timedCurve.info
                    )
                } else if timedCurve.mask == nil {
                    curveMask = cachedCurveAssistMask
                    curveStatus = String(
                        format: cachedCurveAssistMask == nil
                            ? "curveOuterV2 mask nil %.0f %@"
                            : "curveOuterV2 hold nil %.0f %@",
                        curveMS,
                        timedCurve.info
                    )
                } else {
                    curveMask = cachedCurveAssistMask
                    curveStatus = String(
                        format: cachedCurveAssistMask == nil
                            ? "curveOuterV2 uv nil %.0f %@"
                            : "curveOuterV2 hold uvNil %.0f %@",
                        curveMS,
                        timedCurve.info
                    )
                }
            } else if let cachedCurveAssistMask {
                curveMask = cachedCurveAssistMask
                curveStatus = "curveOuterV2 hold/\(curveAssistStride)"
            }

            let stableFillMask: UVLipAlphaMask?
            let cachedCurveForNextFrame: UVLipAlphaMask?
            let curvePrimaryApplied: Bool
            let curvePipelineSource: String
            var pipelineStatus: String
            if let lipAlphaMask = timedFill.mask {
                let smoothedMask = lipAlphaMask.temporallySmoothed(
                    previous: previousMask,
                    response: mouthOpenAmount > 0.10 ? 0.94 : 0.84
                )
                let silhouetteMask = preserveClosedMouthCrease
                    ? smoothedMask.preservedClosedMouthCrease()
                    : smoothedMask
                let featheredMask = self.uvLipMaskFeather.feathered(silhouetteMask, refinement: .uvSegProduct) ?? silhouetteMask
                let baseMask = preserveClosedMouthCrease
                    ? featheredMask.preservedClosedMouthCrease()
                    : featheredMask
                let effectiveCurveMask: UVLipAlphaMask?
                if let curveMask, (curveMaskIsFresh || Self.curveAssistMaskIsReusable(baseMask: baseMask, curveMask: curveMask)) {
                    effectiveCurveMask = curveMask
                } else {
                    effectiveCurveMask = nil
                    if curveMask != nil {
                        curveStatus += " stale"
                    }
                }
                if let effectiveCurveMask,
                   let curveApplied = Self.curvePrimaryLipMask(
                       baseMask: baseMask,
                       curveMask: effectiveCurveMask,
                       mouthOpenAmount: mouthOpenAmount
                   ) {
                    cachedCurveForNextFrame = effectiveCurveMask
                    curvePrimaryApplied = true
                    curvePipelineSource = "curve-primary"
                    stableFillMask = curveApplied.mask
                    pipelineStatus = String(format: " curvePrimary %.2f %@", curveApplied.overlap, curveApplied.reason)
                } else {
                    cachedCurveForNextFrame = cachedCurveAssistMask
                    curvePrimaryApplied = false
                    curvePipelineSource = "uv"
                    stableFillMask = baseMask
                    pipelineStatus = curveMask.map {
                        Self.curvePrimaryLipMaskRejectStatus(
                            baseMask: baseMask,
                            curveMask: $0,
                            mouthOpenAmount: mouthOpenAmount
                        )
                    } ?? " curvePrimaryNone"
                }
            } else {
                curvePrimaryApplied = false
                curvePipelineSource = "none"
                cachedCurveForNextFrame = nil
                stableFillMask = nil
                pipelineStatus = " curvePrimaryNoBase"
            }

            let fusedMask = stableFillMask.flatMap { fill in
                sdfMask.map { contour in
                    Self.fusedSDFUVMask(
                        fillMask: fill,
                        contourMask: contour,
                        previousMask: previousMask,
                        preserveClosedMouthCrease: preserveClosedMouthCrease,
                        mouthOpenAmount: mouthOpenAmount
                    )
                }
            }
            let lipLineMask: UVLipAlphaMask? = nil
            let effectiveFillMask = fusedMask ?? stableFillMask ?? templateFillMask ?? previousMask
            let effectiveLineMask: UVLipAlphaMask? = nil

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard self.lipParsingGeneration == parsingGeneration else {
                    self.isSDFWorkerBusy = false
                    self.isLipParsingInFlight = false
                    self.lastLipParsingStartedAt = 0
                    return
                }
                self.isSDFWorkerBusy = false
                self.isLipParsingInFlight = false
                self.latestLipParsingDurationMS = modelMS
                let fillStatus: String
                if let stableFillMask {
                    fillStatus = String(format: "fill %.2f%%/%d", stableFillMask.nonzeroCoverage * 100, stableFillMask.maximumAlpha)
                } else {
                    fillStatus = "fill nil"
                }
                let sdfStatus: String
                if let sdfMask {
                    sdfStatus = String(format: "sdf %.2f%%/%d", sdfMask.nonzeroCoverage * 100, sdfMask.maximumAlpha)
                } else {
                    sdfStatus = "sdf nil"
                }
                let effectiveStatus: String
                if let effectiveFillMask {
                    effectiveStatus = String(format: "fuse %.2f%%/%d", effectiveFillMask.nonzeroCoverage * 100, effectiveFillMask.maximumAlpha)
                } else {
                    effectiveStatus = "fuse nil"
                }
                let lineStatus: String
                if let effectiveLineMask {
                    lineStatus = String(format: "line %.2f%%/%d", effectiveLineMask.nonzeroCoverage * 100, effectiveLineMask.maximumAlpha)
                } else {
                    lineStatus = "line nil"
                }
                self.latestLipParsingBreakdownText = String(
                    format: "uv prep %.0f pred %.0f dec %.0f raw %.0f post %.0f %@ %@ %@  sdf prep %.0f pred %.0f dec %.0f raw %.0f post %.0f %@  %@ %@ %@ %@",
                    timedFill.inputMS,
                    timedFill.predictionMS,
                    timedFill.decodeMS,
                    timedFill.decodeRawMS,
                    timedFill.decodePostMS,
                    timedFill.outputDataType,
                    "\(timedFill.decodeInfo)\(pipelineStatus)",
                    fillStatus,
                    timedSDF.inputMS,
                    timedSDF.predictionMS,
                    timedSDF.decodeMS,
                    timedSDF.decodeRawMS,
                    timedSDF.decodePostMS,
                    timedSDF.outputDataType,
                    sdfStatus,
                    effectiveStatus,
                    lineStatus,
                    curveStatus,
                    templateStatus
                )

                guard
                    self.currentMakeupVisible,
                    self.shouldAcceptUVLipStudentResult,
                    let effectiveFillMask
                else {
                    self.emitPipelineConsoleLog(trigger: "sdf-branch-empty")
                    self.emitMaskGeometryConsoleLog(trigger: "sdf-branch-empty", mask: effectiveFillMask)
                    if self.currentMakeupVisible, self.shouldAcceptUVLipStudentResult,
                       let heldMask = self.latestUVLipAlphaMask {
                        self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                        self.latestUVLipAlphaMask = heldMask
                        self.latestUVLipSmoothingSourceMask = heldMask
                        let heldLineMask = self.latestUVLipLineRenderMask
                        self.latestUVLipLineRenderMask = heldLineMask
                        self.latestUVLipLineDebugMask = self.currentLook?.isLipLineDebug == true ? heldLineMask : nil
                        self.latestLipParsingBreakdownText = (self.latestLipParsingBreakdownText ?? "") + " hold-prev"
                        self.emitPipelineConsoleLog(trigger: "sdf-branch-hold")
                        self.emitMaskGeometryConsoleLog(trigger: "sdf-branch-hold", mask: heldMask)
                        self.needsTextureRefresh = true
                        self.updateMaterialTexture()
                        self.clearPreciseLipMakeupLayer()
                        self.lipParsingOverlayLayer.isHidden = true
                        self.updateLipOverlayGeometry(nil)
                        return
                    }
                    self.hideStaleLipParsingOverlay()
                    return
                }

                self.latestCurveAssistMask = cachedCurveForNextFrame
                if curvePrimaryApplied {
                    self.curveRejectStreak = 0
                    self.curveBackoffFramesRemaining = 0
                } else if curveStatus.contains("curveOuterV2 ok") || curveStatus.contains("curveOuterV2 hold") || curveStatus.contains("curveOuterV2 uv nil") || curveStatus.contains("curveOuterV2 mask nil") {
                    self.curveRejectStreak += 1
                    if self.curveRejectStreak >= 3 {
                        self.curveBackoffFramesRemaining = 18
                        self.curveRejectStreak = 0
                    }
                }

                let displayedMask: UVLipAlphaMask
                let lockStatus: String
                if shouldBypassUVLock {
                    self.uvLipMaskLock.reset()
                    displayedMask = effectiveFillMask
                    lockStatus = String(format: " uvLock open%.2f", mouthOpenAmount)
                } else {
                    let lockResult = self.uvLipMaskLock.resolve(candidate: effectiveFillMask)
                    displayedMask = lockResult.mask
                    lockStatus = " uvLock \(lockResult.decision.statusText)"
                }

                let displayedLineMask: UVLipAlphaMask?
                let lineLockStatus: String
                if let effectiveLineMask {
                    let clippedLineMask = effectiveLineMask.clipped(
                        to: displayedMask.expanded(
                            threshold: 24,
                            radius: 0,
                            blurRadius: 0
                        )
                    )
                    if shouldBypassUVLock {
                        self.uvLipLineFusedMaskLock.reset()
                        displayedLineMask = clippedLineMask
                        lineLockStatus = String(format: " lineLock open%.2f", mouthOpenAmount)
                    } else {
                        let lineLockResult = self.uvLipLineFusedMaskLock.resolve(candidate: clippedLineMask)
                        displayedLineMask = lineLockResult.mask
                        lineLockStatus = " lineLock \(lineLockResult.decision.statusText)"
                    }
                } else {
                    self.uvLipLineFusedMaskLock.reset()
                    displayedLineMask = nil
                    lineLockStatus = ""
                }

                self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                self.latestUVLipAlphaMask = displayedMask
                self.latestUVLipSmoothingSourceMask = displayedMask
                self.latestUVLipColorSupportMask = nil
                self.latestUVLipLineRenderMask = displayedLineMask
                self.latestUVLipLineDebugMask = self.currentLook?.isLipLineDebug == true ? displayedLineMask : nil
                self.latestUVLipLineOnlyDebugMask = nil
                self.latestUVLipLineColorDebugMask = nil
                self.latestUVLipLineFusedDebugMask = nil
                self.latestLipRenderDurationMS = nil
                self.updateLipOverlayGeometry(nil)
                let finalSource = fusedMask == nil && templateFillMask != nil ? "template-first" : "sdf+uv"
                self.latestLipParsingBreakdownText = (self.latestLipParsingBreakdownText ?? "") + String(
                    format: " final %.2f%%/%d%@%@ finalSrc %@",
                    displayedMask.nonzeroCoverage * 100,
                    displayedMask.maximumAlpha,
                    lockStatus,
                    lineLockStatus,
                    finalSource
                )
                self.emitPipelineConsoleLog(trigger: "sdf-branch-final")
                self.emitMaskGeometryConsoleLog(trigger: "sdf-branch-final", mask: displayedMask)
                self.needsTextureRefresh = true
                self.updateMaterialTexture()
                self.clearPreciseLipMakeupLayer()
                self.lipParsingOverlayLayer.isHidden = true
            }
        }
    }

    private func primeLipSDFHotStartIfPossible() {
        guard
            currentLipMode == .lipSDFRefinement,
            !usesForcedRawUVPipeline,
            currentMakeupVisible,
            let _ = sceneView.session.currentFrame
        else {
            return
        }

        if let latestUVLipAlphaMask {
            latestLipParsingUpdatedAt = CACurrentMediaTime()
            latestUVLipLineDebugMask = currentLook?.isLipLineDebug == true ? latestUVLipLineRenderMask : nil
            latestLipParsingBreakdownText = "hotStart carried"
            needsTextureRefresh = true
            updateMaterialTexture(forceSynchronous: true)
            emitPipelineConsoleLog(trigger: "sdf-hot-start")
            emitMaskGeometryConsoleLog(trigger: "sdf-hot-start", mask: latestUVLipAlphaMask)
            return
        }
        latestLipParsingBreakdownText = "hotStart waiting steady-state"
    }

    private func primeRawUVDebugHotStartIfPossible() {
        guard
            currentLipMode.usesUVSegStudent,
            usesForcedRawUVPipeline,
            currentMakeupVisible,
            let currentFrame = sceneView.session.currentFrame,
            let faceAnchor = latestFaceAnchor,
            rawUVHotStartModel.isReady,
            !isRawUVHotStartInFlight
        else {
            return
        }

        isRawUVHotStartInFlight = true
        if !didLogStartupHotStart {
            didLogStartupHotStart = true
            logTryOnStartupEvent(
                stage: "hot-start-begin",
                details: "uvReady=\(rawUVHotStartModel.isReady ? 1 : 0) mouthOpen=\(String(format: "%.3f", latestMouthOpenAmount))"
            )
        }
        let parsingGeneration = lipParsingGeneration

        guard let request = rawUVHotStartModel.makeRequest(
            pixelBuffer: currentFrame.capturedImage,
            arFrame: currentFrame,
            faceAnchor: faceAnchor,
            viewportSize: viewportSize
        ) else {
            isRawUVHotStartInFlight = false
            return
        }

        rawUVHotStartQueue.async { [weak self, rawUVHotStartModel, request] in
            let timedUV = rawUVHotStartModel.timedLipAlphaMask(request: request)
            guard let baseMask = timedUV.mask else {
                DispatchQueue.main.async { [weak self] in
                    self?.isRawUVHotStartInFlight = false
                }
                return
            }

            let mouthOpenAmount = self?.latestMouthOpenAmount ?? 0
            let preserveClosedMouthCrease = mouthOpenAmount < 0.12
            let hotMask = Self.preparedUVFillMask(
                fillMask: baseMask,
                previousMask: nil,
                contourBase: baseMask,
                preserveClosedMouthCrease: preserveClosedMouthCrease,
                mouthOpenAmount: mouthOpenAmount
            )

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.isRawUVHotStartInFlight = false
                let existingMask = self.latestUVLipAlphaMask
                let hasUsableDisplayedMask = (existingMask?.maximumAlpha ?? 0) >= 24
                    && (existingMask?.nonzeroCoverage ?? 0) > 0.001
                guard
                    self.lipParsingGeneration == parsingGeneration,
                    self.currentLipMode.usesUVSegStudent,
                    self.usesForcedRawUVPipeline,
                    self.currentMakeupVisible,
                    !hasUsableDisplayedMask
                else {
                    self.lastLipParsingStartedAt = 0
                    return
                }

                self.latestLipParsingUpdatedAt = CACurrentMediaTime()
                self.latestUVLipAlphaMask = hotMask
                self.latestUVLipSmoothingSourceMask = baseMask
                self.latestUVLipColorSupportMask = nil
                self.latestUVLipLineRenderMask = nil
                self.latestUVLipLineDebugMask = nil
                self.latestUVLipLineOnlyDebugMask = nil
                self.latestUVLipLineColorDebugMask = nil
                self.latestUVLipLineFusedDebugMask = nil
                self.latestLipParsingBreakdownText = String(
                    format: "hotStart raw-uv prep %.0f pred %.0f dec %.0f raw %.0f post %.0f %@ uv %.2f%%/%d",
                    timedUV.inputMS,
                    timedUV.predictionMS,
                    timedUV.decodeMS,
                    timedUV.decodeRawMS,
                    timedUV.decodePostMS,
                    timedUV.outputDataType,
                    hotMask.nonzeroCoverage * 100,
                    hotMask.maximumAlpha
                )
                self.logTryOnStartupEvent(
                    stage: "hot-start-ready",
                    details: String(
                        format: "pred=%.0f decode=%.0f cov=%.2f max=%d",
                        timedUV.predictionMS,
                        timedUV.decodeMS,
                        hotMask.nonzeroCoverage * 100,
                        hotMask.maximumAlpha
                    )
                )
                self.renderFastRawUVShadePreviewIfPossible()
                self.needsTextureRefresh = true
                self.updateMaterialTexture()
                self.emitPipelineConsoleLog(trigger: "raw-uv-hot-start")
                self.emitMaskGeometryConsoleLog(trigger: "raw-uv-hot-start", mask: hotMask)
            }
        }
    }

    private static func fusedSDFUVMask(
        fillMask: UVLipAlphaMask,
        contourMask: UVLipAlphaMask,
        previousMask: UVLipAlphaMask?,
        preserveClosedMouthCrease: Bool,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        let contourBase = preparedSDFContourMask(
            contourMask: contourMask,
            previousMask: previousMask,
            preserveClosedMouthCrease: preserveClosedMouthCrease,
            mouthOpenAmount: mouthOpenAmount
        )
        let uvShape = preparedUVFillMask(
            fillMask: fillMask,
            previousMask: previousMask,
            contourBase: contourBase,
            preserveClosedMouthCrease: preserveClosedMouthCrease,
            mouthOpenAmount: mouthOpenAmount
        )
        let contourBlend = contourBase
            .clipped(
                to: uvShape.expanded(
                    threshold: 24,
                    radius: 1,
                    blurRadius: 1
                )
            )
            .expanded(
                threshold: 24,
                radius: 0,
                blurRadius: 1
            )
        let useContourBlend = contourBase.maximumAlpha > 80
            && contourBlend.nonzeroCoverage > max(uvShape.nonzeroCoverage * 0.06, 0.0008)
        let fusedFill = useContourBlend
            ? uvShape.merged(with: contourBlend)
            : uvShape
        return fusedFill.productAlphaSmoothedForMakeup(edgeThreshold: 32)
    }

    private static func sdfLipLineMask(
        fillMask: UVLipAlphaMask,
        contourMask: UVLipAlphaMask,
        previousFillMask: UVLipAlphaMask?,
        previousLineMask: UVLipAlphaMask?,
        preserveClosedMouthCrease: Bool,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        let contourBase = preparedSDFContourMask(
            contourMask: contourMask,
            previousMask: previousFillMask,
            preserveClosedMouthCrease: preserveClosedMouthCrease,
            mouthOpenAmount: mouthOpenAmount
        )
        let uvShape = preparedUVFillMask(
            fillMask: fillMask,
            previousMask: previousFillMask,
            contourBase: contourBase,
            preserveClosedMouthCrease: preserveClosedMouthCrease,
            mouthOpenAmount: mouthOpenAmount
        )
        let uvBaseLine = uvShape
            .outerSilhouetteOutline(
                threshold: 72,
                widthPixels: mouthOpenAmount > 0.12 ? 1 : 2
            )
            .smoothLipLineSilhouette(threshold: 72, radius: 1)
        let sdfBoost = contourBase
            .outerSilhouetteOutline(
                threshold: 72,
                widthPixels: 1
            )
            .clipped(
                to: uvBaseLine.expanded(
                    threshold: 24,
                    radius: 1,
                    blurRadius: 0
                )
            )
        let useSDFBoost = contourBase.maximumAlpha > 96
            && sdfBoost.nonzeroCoverage > max(uvBaseLine.nonzeroCoverage * 0.08, 0.0008)
        let composedOuterLine = useSDFBoost
            ? uvBaseLine.merged(with: sdfBoost)
            : uvBaseLine
        let innerLine: UVLipAlphaMask?
        if mouthOpenAmount > 0.08,
           let holeMask = uvShape.enclosedHoleMask(
               threshold: 24,
               expansionRadius: 0,
               blurRadius: 0
           ) {
            innerLine = holeMask
                .outerSilhouetteOutline(threshold: 72, widthPixels: 1)
                .clipped(
                    to: uvShape.expanded(
                        threshold: 24,
                        radius: 0,
                        blurRadius: 0
                    )
                )
        } else {
            innerLine = nil
        }
        let composedLine = innerLine.map { composedOuterLine.merged(with: $0) } ?? composedOuterLine
        return composedLine
            .temporallySmoothed(previous: previousLineMask, response: mouthOpenAmount > 0.10 ? 0.82 : 0.68)
            .clipped(
                to: uvShape.expanded(
                    threshold: 24,
                    radius: 1,
                    blurRadius: 0
                )
            )
            .softenedStroke(expansionRadius: 0, blurRadius: 1)
    }

    private static func preparedTemplateWarmStartMask(
        fillMask: UVLipAlphaMask,
        previousMask: UVLipAlphaMask?,
        preserveClosedMouthCrease: Bool,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        preparedUVFillMask(
            fillMask: fillMask,
            previousMask: previousMask,
            contourBase: fillMask,
            preserveClosedMouthCrease: preserveClosedMouthCrease,
            mouthOpenAmount: mouthOpenAmount
        )
    }

    private static func templateWarmStartLineMask(
        fillMask: UVLipAlphaMask,
        previousLineMask: UVLipAlphaMask?,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        let outerLine = fillMask
            .outerSilhouetteOutline(
                threshold: 72,
                widthPixels: mouthOpenAmount > 0.12 ? 1 : 2
            )
            .smoothLipLineSilhouette(threshold: 72, radius: 1)
        let innerLine: UVLipAlphaMask?
        if mouthOpenAmount > 0.08,
           let holeMask = fillMask.enclosedHoleMask(
               threshold: 24,
               expansionRadius: 0,
               blurRadius: 0
           ) {
            innerLine = holeMask.outerSilhouetteOutline(threshold: 72, widthPixels: 1)
        } else {
            innerLine = nil
        }
        let composedLine = innerLine.map { outerLine.merged(with: $0) } ?? outerLine
        return composedLine.temporallySmoothed(
            previous: previousLineMask,
            response: mouthOpenAmount > 0.10 ? 0.76 : 0.62
        )
    }

    private static func preparedUVFillMask(
        fillMask: UVLipAlphaMask,
        previousMask: UVLipAlphaMask?,
        contourBase: UVLipAlphaMask,
        preserveClosedMouthCrease: Bool,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        let smoothedFill = fillMask.temporallySmoothed(
            previous: previousMask,
            response: mouthOpenAmount > 0.10 ? 0.90 : 0.74
        )
        var uvShape = smoothedFill
            .productAlphaSmoothedForMakeup(edgeThreshold: 40)
            .expanded(
                threshold: 40,
                radius: 0,
                blurRadius: 0
            )
        if preserveClosedMouthCrease {
            uvShape = uvShape.preservedClosedMouthCrease()
        }
        if mouthOpenAmount > 0.08,
           let holeMask = uvShape.enclosedHoleMask(
               threshold: 24,
               expansionRadius: mouthOpenAmount > 0.22 ? 1 : 0,
               blurRadius: 1
           ) {
            uvShape = uvShape.subtracting(holeMask)
        }
        return uvShape.productAlphaSmoothedForMakeup(edgeThreshold: 40)
    }

    private static func preparedSDFContourMask(
        contourMask: UVLipAlphaMask,
        previousMask: UVLipAlphaMask?,
        preserveClosedMouthCrease: Bool,
        mouthOpenAmount: CGFloat
    ) -> UVLipAlphaMask {
        let smoothedContour = contourMask.temporallySmoothed(
            previous: previousMask,
            response: mouthOpenAmount > 0.10 ? 0.86 : 0.66
        )
        var contourBase = smoothedContour
            .productAlphaSmoothedForMakeup(edgeThreshold: 24)
            .expanded(
                threshold: 24,
                radius: mouthOpenAmount > 0.12 ? 0 : 1,
                blurRadius: 1
            )
        if preserveClosedMouthCrease {
            contourBase = contourBase.preservedClosedMouthCrease()
        }
        if mouthOpenAmount > 0.08,
           let holeMask = contourBase.enclosedHoleMask(
               threshold: 24,
               expansionRadius: mouthOpenAmount > 0.22 ? 1 : 0,
               blurRadius: 1
           ) {
            contourBase = contourBase.subtracting(holeMask)
        }
        return contourBase.productAlphaSmoothedForMakeup(edgeThreshold: 24)
    }

    private func updateLipParsingOverlayFrame(to frame: CGRect) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            self.latestLipOverlayFrame = frame

            let nextFrame: CGRect
            if let smoothedLipOverlayFrame {
                nextFrame = smoothedLipOverlayFrame.interpolated(to: frame, amount: 0.93)
            } else {
                nextFrame = frame
            }

            self.smoothedLipOverlayFrame = nextFrame

            CATransaction.begin()
            CATransaction.setDisableActions(true)
            if self.currentMakeupVisible, self.shouldDisplayScreenLipOverlay, self.preciseLipMakeupLayer.contents != nil {
                let now = CACurrentMediaTime()
                if let screenFrame = self.screenLipMaskTracker.trackedFrame(for: frame, now: now) {
                    self.latestScreenLipDisplayFrame = screenFrame
                    self.setPreciseLipMakeupFrame(screenFrame)
                    self.preciseLipMakeupLayer.isHidden = false
                } else if self.currentLipMode == .autoFusion, !self.screenLipMaskTracker.hasRecentMask(now: now) {
                    self.latestScreenLipDisplayFrame = nil
                    self.preciseLipMakeupLayer.isHidden = true
                } else {
                    self.latestScreenLipDisplayFrame = nextFrame
                    self.setPreciseLipMakeupFrame(nextFrame)
                    self.preciseLipMakeupLayer.isHidden = false
                }
            }

            guard
                self.currentMakeupVisible,
                self.currentShowFaceMesh,
                self.lipParsingOverlayLayer.contents != nil
            else {
                self.lipParsingOverlayLayer.isHidden = true
                CATransaction.commit()
                return
            }

            self.lipParsingOverlayLayer.frame = nextFrame
            self.lipParsingOverlayLayer.isHidden = false
            CATransaction.commit()
        }
    }

    private func updateScreenLipMakeupLayer(with lipMask: ARKitLipMask) {
        guard
            currentMakeupVisible,
            shouldDisplayScreenLipOverlay,
            let look = currentLook
        else {
            clearPreciseLipMakeupLayer()
            return
        }

        let now = CACurrentMediaTime()
        guard screenLipMaskLooksUsable(lipMask) else {
            if updateTrackedScreenLipMakeupFrame(now: now) {
                return
            }

            clearPreciseLipMakeupLayer()
            return
        }

        latestScreenLipMask = lipMask
        latestScreenLipMaskUpdatedAt = now
        let targetFrame = screenLipMaskTracker.acceptMaskFrame(
            lipMask.frame,
            regionOfInterest: latestLipOverlayFrame,
            now: now
        )
        latestScreenLipDisplayFrame = targetFrame
        let renderStart = CACurrentMediaTime()
        let lipColor = boostedLipUIColor(from: look)
        guard let tintedLipImage = lipMask.tintedImage(color: lipColor) else {
            preciseLipMakeupLayer.isHidden = true
            return
        }
        latestLipRenderDurationMS = (CACurrentMediaTime() - renderStart) * 1000
        setPreciseLipMakeupFrame(targetFrame)
        preciseLipMakeupLayer.contentsScale = window?.screen.scale ?? UIScreen.main.scale
        preciseLipMakeupLayer.backgroundColor = nil
        preciseLipMakeupLayer.contents = tintedLipImage
        preciseLipMaskLayer.contents = nil
        preciseLipMakeupLayer.isHidden = false
    }

    private func updateTrackedScreenLipMakeupFrame(now: CFTimeInterval = CACurrentMediaTime()) -> Bool {
        guard
            currentMakeupVisible,
            shouldDisplayScreenLipOverlay,
            preciseLipMakeupLayer.contents != nil,
            let latestLipOverlayFrame,
            let trackedFrame = screenLipMaskTracker.trackedFrame(for: latestLipOverlayFrame, now: now)
        else {
            return false
        }

        latestScreenLipDisplayFrame = trackedFrame
        setPreciseLipMakeupFrame(trackedFrame)
        preciseLipMakeupLayer.isHidden = false
        return true
    }

    private func screenLipMaskLooksUsable(_ lipMask: ARKitLipMask) -> Bool {
        guard lipMask.width > 0, lipMask.height > 0, lipMask.alpha.count == lipMask.width * lipMask.height else {
            return false
        }

        let activePixelCount = lipMask.alpha.reduce(0) { count, alpha in
            count + (alpha > 10 ? 1 : 0)
        }
        let coverage = CGFloat(activePixelCount) / CGFloat(max(lipMask.width * lipMask.height, 1))
        let maximumAlpha = lipMask.alpha.max() ?? 0

        return coverage >= 0.006
            && coverage <= 0.66
            && maximumAlpha >= 16
    }

    private func setPreciseLipMakeupFrame(_ frame: CGRect) {
        preciseLipMakeupLayer.frame = frame
        preciseLipMaskLayer.frame = preciseLipMakeupLayer.bounds
    }

    private func clearPreciseLipMakeupLayer() {
        preciseLipMaskLayer.contents = nil
        preciseLipMakeupLayer.contents = nil
        preciseLipMakeupLayer.backgroundColor = nil
        preciseLipMakeupLayer.isHidden = true
    }

    private func boostedLipUIColor(from look: MakeupLook) -> UIColor {
        let color = look.lip.uiColor(intensity: currentIntensity)
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard color.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return color
        }

        return UIColor(
            red: red,
            green: green,
            blue: blue,
            alpha: min(max(alpha * 1.08, 0), 1)
        )
    }

    private func refreshScreenLipMakeupLayer() {
        guard shouldDisplayScreenLipOverlay else {
            clearPreciseLipMakeupLayer()
            return
        }

        guard let latestScreenLipMask else {
            preciseLipMakeupLayer.isHidden = true
            return
        }

        updateScreenLipMakeupLayer(with: latestScreenLipMask)
    }

    private func hasRecentScreenLipMask(now: CFTimeInterval = CACurrentMediaTime()) -> Bool {
        guard latestScreenLipMask != nil, let latestScreenLipMaskUpdatedAt else { return false }
        return now - latestScreenLipMaskUpdatedAt <= screenLipMaskFreezeDuration
    }

    private func updateLipOverlayVisibility() {
        lipOverlayNode?.isHidden = !(currentMakeupVisible && currentLipMode.usesGPULipOverlay && latestLipOverlayGeometry != nil)
    }

    private func updateLipOverlayGeometry(_ geometry: SCNGeometry?) {
        let hadOverlayGeometry = latestLipOverlayGeometry != nil
        latestLipOverlayGeometry = geometry
        geometry?.firstMaterial = lipOverlayMaterial
        lipOverlayNode?.geometry = geometry
        if currentLipMode.usesGPULipOverlay, hadOverlayGeometry != (geometry != nil) {
            updateMaterialTexture()
        }
        updateLipOverlayVisibility()
    }

    private func refreshLatestLipParsingImage() {
        guard
            currentMakeupVisible,
            currentShowFaceMesh,
            let latestFaceParsingMask,
            let image = faceParsingCompositor.makeDebugImage(from: latestFaceParsingMask) ?? latestFaceParsingMask.debugImage()
        else {
            lipParsingOverlayLayer.isHidden = true
            return
        }

        lipParsingOverlayLayer.contents = image
        lipParsingOverlayLayer.isHidden = false
    }

    private func activeParsingROI(frame: ARFrame, faceAnchor: ARFaceAnchor) -> CGRect? {
        if currentLipMode == .autoFusion, activeAutoLipRenderPath.usesScreenOverlay, lipRefinementSmallModel.isReady {
            return projectedMouthROI(frame: frame, faceAnchor: faceAnchor)
        }

        if currentLipMode.usesLipROIModel, activeLipRefinementModel?.isReady == true {
            return projectedMouthROI(frame: frame, faceAnchor: faceAnchor)
        }

        return projectedFaceROI(frame: frame, faceAnchor: faceAnchor)
    }

    private func lipProjectionColor(for look: MakeupLook) -> SIMD4<Float> {
        SIMD4<Float>(
            Float(look.lip.red),
            Float(look.lip.green),
            Float(look.lip.blue),
            Float(min(max(look.lip.alpha * CGFloat(currentIntensity) * 1.08, 0), 1))
        )
    }

    private func lipMeshProjectionVertices(
        frame: ARFrame,
        faceAnchor: ARFaceAnchor,
        maskFrame: CGRect
    ) -> [LipMaskFaceMeshProjectionVertex]? {
        let viewSize = viewportSize
        guard
            viewSize.width > 1,
            viewSize.height > 1,
            maskFrame.width > 1,
            maskFrame.height > 1
        else {
            return nil
        }

        let faceGeometry = faceAnchor.geometry
        let vertices = faceGeometry.vertices
        let textureCoordinates = faceGeometry.textureCoordinates
        let triangleIndices = faceGeometry.triangleIndices
        guard
            vertices.count == textureCoordinates.count,
            vertices.count > 80,
            triangleIndices.count >= 3
        else {
            return nil
        }

        var screenPoints = [CGPoint](repeating: .zero, count: vertices.count)
        var validScreenPoints = [Bool](repeating: false, count: vertices.count)

        for index in vertices.indices {
            let vertex = vertices[index]
            let worldPoint4 = simd_mul(
                faceAnchor.transform,
                simd_float4(vertex.x, vertex.y, vertex.z, 1)
            )
            let worldPoint = simd_float3(worldPoint4.x, worldPoint4.y, worldPoint4.z)
            let projected = frame.camera.projectPoint(
                worldPoint,
                orientation: .portrait,
                viewportSize: viewSize
            )

            guard
                projected.x.isFinite,
                projected.y.isFinite,
                projected.x > -viewSize.width * 0.35,
                projected.x < viewSize.width * 1.35,
                projected.y > -viewSize.height * 0.35,
                projected.y < viewSize.height * 1.35
            else {
                continue
            }

            screenPoints[index] = projected
            validScreenPoints[index] = true
        }

        let lipUVBounds = CGRect(x: 0.34, y: 0.52, width: 0.32, height: 0.24)
        var projectionVertices: [LipMaskFaceMeshProjectionVertex] = []
        projectionVertices.reserveCapacity(triangleIndices.count)

        for offset in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let i0 = Int(triangleIndices[offset])
            let i1 = Int(triangleIndices[offset + 1])
            let i2 = Int(triangleIndices[offset + 2])

            guard
                vertices.indices.contains(i0),
                vertices.indices.contains(i1),
                vertices.indices.contains(i2),
                validScreenPoints[i0],
                validScreenPoints[i1],
                validScreenPoints[i2]
            else {
                continue
            }

            let uv0 = textureCoordinates[i0]
            let uv1 = textureCoordinates[i1]
            let uv2 = textureCoordinates[i2]
            guard
                uv0.x.isFinite,
                uv0.y.isFinite,
                uv1.x.isFinite,
                uv1.y.isFinite,
                uv2.x.isFinite,
                uv2.y.isFinite
            else {
                continue
            }

            let uvCenter = CGPoint(
                x: CGFloat((uv0.x + uv1.x + uv2.x) / 3),
                y: CGFloat((uv0.y + uv1.y + uv2.y) / 3)
            )
            guard lipUVBounds.contains(uvCenter) else { continue }

            let area = abs(
                (screenPoints[i0].x * (screenPoints[i1].y - screenPoints[i2].y))
                    + (screenPoints[i1].x * (screenPoints[i2].y - screenPoints[i0].y))
                    + (screenPoints[i2].x * (screenPoints[i0].y - screenPoints[i1].y))
            ) * 0.5
            guard area > 0.04 else { continue }

            projectionVertices.append(
                lipMeshProjectionVertex(facePoint: vertices[i0], screenPoint: screenPoints[i0], maskFrame: maskFrame)
            )
            projectionVertices.append(
                lipMeshProjectionVertex(facePoint: vertices[i1], screenPoint: screenPoints[i1], maskFrame: maskFrame)
            )
            projectionVertices.append(
                lipMeshProjectionVertex(facePoint: vertices[i2], screenPoint: screenPoints[i2], maskFrame: maskFrame)
            )
        }

        guard projectionVertices.count >= 36 else { return nil }
        return projectionVertices
    }

    private func lipMeshProjectionVertex(
        facePoint: vector_float3,
        screenPoint: CGPoint,
        maskFrame: CGRect
    ) -> LipMaskFaceMeshProjectionVertex {
        let maskU = (screenPoint.x - maskFrame.minX) / maskFrame.width
        let maskV = (screenPoint.y - maskFrame.minY) / maskFrame.height
        return LipMaskFaceMeshProjectionVertex(
            position: SIMD3<Float>(facePoint.x, facePoint.y, facePoint.z),
            maskUV: SIMD2<Float>(Float(maskU), Float(maskV))
        )
    }

    private func projectedMouthROI(frame: ARFrame, faceAnchor: ARFaceAnchor) -> CGRect? {
        let viewSize = viewportSize
        guard viewSize.width > 1, viewSize.height > 1 else { return nil }

        let geometry = faceAnchor.geometry
        let vertices = geometry.vertices
        let textureCoordinates = geometry.textureCoordinates
        guard vertices.count > 100, vertices.count == textureCoordinates.count else { return nil }

        var projectedPoints: [CGPoint] = []
        projectedPoints.reserveCapacity(96)
        let lipUVBounds = CGRect(x: 0.34, y: 0.52, width: 0.32, height: 0.24).insetBy(dx: -0.035, dy: -0.045)

        for index in vertices.indices {
            let uv = textureCoordinates[index]
            let uvPoint = CGPoint(x: CGFloat(uv.x), y: CGFloat(uv.y))
            guard
                uvPoint.x.isFinite,
                uvPoint.y.isFinite,
                lipUVBounds.contains(uvPoint)
            else {
                continue
            }

            let vertex = vertices[index]
            let worldPoint4 = simd_mul(
                faceAnchor.transform,
                simd_float4(vertex.x, vertex.y, vertex.z, 1)
            )
            let worldPoint = simd_float3(worldPoint4.x, worldPoint4.y, worldPoint4.z)
            let projected = frame.camera.projectPoint(
                worldPoint,
                orientation: .portrait,
                viewportSize: viewSize
            )

            guard
                projected.x.isFinite,
                projected.y.isFinite,
                projected.x > -viewSize.width * 0.18,
                projected.x < viewSize.width * 1.18,
                projected.y > -viewSize.height * 0.18,
                projected.y < viewSize.height * 1.18
            else {
                continue
            }

            projectedPoints.append(projected)
        }

        guard projectedPoints.count >= 12 else { return nil }

        let minPX = projectedPoints.map(\.x).min() ?? 0
        let maxPX = projectedPoints.map(\.x).max() ?? 0
        let minPY = projectedPoints.map(\.y).min() ?? 0
        let maxPY = projectedPoints.map(\.y).max() ?? 0
        let mouthBounds = CGRect(x: minPX, y: minPY, width: maxPX - minPX, height: maxPY - minPY)
        guard mouthBounds.width > 32, mouthBounds.height > 18 else { return nil }

        let targetAspect = CGFloat(lipCurveModel.inputWidth) / CGFloat(max(lipCurveModel.inputHeight, 1))
        var cropWidth = max(mouthBounds.width * 1.28, 112)
        var cropHeight = max(mouthBounds.height * 1.54, 72)
        if cropWidth / cropHeight < targetAspect {
            cropWidth = cropHeight * targetAspect
        } else {
            cropHeight = cropWidth / targetAspect
        }

        cropWidth = min(cropWidth, viewSize.width * 0.78)
        cropHeight = min(cropHeight, viewSize.height * 0.34)

        var originX = mouthBounds.midX - cropWidth * 0.5
        var originY = mouthBounds.midY - cropHeight * 0.45
        originX = min(max(originX, 0), max(viewSize.width - cropWidth, 0))
        originY = min(max(originY, 0), max(viewSize.height - cropHeight, 0))

        return CGRect(x: originX, y: originY, width: cropWidth, height: cropHeight)
            .intersection(CGRect(origin: .zero, size: viewSize))
    }

    private func projectedFaceROI(frame: ARFrame, faceAnchor: ARFaceAnchor) -> CGRect? {
        let viewSize = viewportSize
        guard viewSize.width > 1, viewSize.height > 1 else { return nil }

        let vertices = faceAnchor.geometry.vertices
        var projectedPoints: [CGPoint] = []
        projectedPoints.reserveCapacity(vertices.count)

        for index in vertices.indices {
            let vertex = vertices[index]
            let worldPoint4 = simd_mul(
                faceAnchor.transform,
                simd_float4(vertex.x, vertex.y, vertex.z, 1)
            )
            let worldPoint = simd_float3(worldPoint4.x, worldPoint4.y, worldPoint4.z)
            let projected = frame.camera.projectPoint(
                worldPoint,
                orientation: .portrait,
                viewportSize: viewSize
            )

            guard
                projected.x.isFinite,
                projected.y.isFinite,
                projected.x > -viewSize.width * 0.25,
                projected.x < viewSize.width * 1.25,
                projected.y > -viewSize.height * 0.25,
                projected.y < viewSize.height * 1.25
            else {
                continue
            }

            projectedPoints.append(projected)
        }

        guard projectedPoints.count >= 80 else { return nil }

        let minX = projectedPoints.map(\.x).min() ?? 0
        let maxX = projectedPoints.map(\.x).max() ?? 0
        let minY = projectedPoints.map(\.y).min() ?? 0
        let maxY = projectedPoints.map(\.y).max() ?? 0
        let faceBounds = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        guard faceBounds.width > 90, faceBounds.height > 110 else { return nil }

        let side = min(
            max(faceBounds.width * 1.52, faceBounds.height * 1.28, 220),
            min(viewSize.width * 0.96, viewSize.height * 0.74)
        )
        var originX = faceBounds.midX - side * 0.5
        var originY = faceBounds.midY - side * 0.49
        originX = min(max(originX, 0), max(viewSize.width - side, 0))
        originY = min(max(originY, 0), max(viewSize.height - side, 0))

        return CGRect(x: originX, y: originY, width: side, height: side)
            .intersection(CGRect(origin: .zero, size: viewSize))
    }

    private func faceUVProjectionMap(frame: ARFrame, faceAnchor: ARFaceAnchor) -> ARKitFaceUVProjectionMap? {
        let viewSize = viewportSize
        guard viewSize.width > 1, viewSize.height > 1 else { return nil }

        let faceGeometry = faceAnchor.geometry
        let vertices = faceGeometry.vertices
        let textureCoordinates = faceGeometry.textureCoordinates
        let triangleIndices = faceGeometry.triangleIndices
        guard
            vertices.count == textureCoordinates.count,
            vertices.count > 80,
            triangleIndices.count >= 3
        else {
            return nil
        }

        var screenPoints = [CGPoint](repeating: .zero, count: vertices.count)
        var validScreenPoints = [Bool](repeating: false, count: vertices.count)

        for index in vertices.indices {
            let vertex = vertices[index]
            let worldPoint4 = simd_mul(
                faceAnchor.transform,
                simd_float4(vertex.x, vertex.y, vertex.z, 1)
            )
            let worldPoint = simd_float3(worldPoint4.x, worldPoint4.y, worldPoint4.z)
            let projected = frame.camera.projectPoint(
                worldPoint,
                orientation: .portrait,
                viewportSize: viewSize
            )

            guard
                projected.x.isFinite,
                projected.y.isFinite,
                projected.x > -viewSize.width * 0.35,
                projected.x < viewSize.width * 1.35,
                projected.y > -viewSize.height * 0.35,
                projected.y < viewSize.height * 1.35
            else {
                continue
            }

            screenPoints[index] = projected
            validScreenPoints[index] = true
        }

        var triangles: [ARKitFaceUVProjectionMap.Triangle] = []
        triangles.reserveCapacity(triangleIndices.count / 3)

        for offset in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let i0 = Int(triangleIndices[offset])
            let i1 = Int(triangleIndices[offset + 1])
            let i2 = Int(triangleIndices[offset + 2])

            guard
                vertices.indices.contains(i0),
                vertices.indices.contains(i1),
                vertices.indices.contains(i2),
                validScreenPoints[i0],
                validScreenPoints[i1],
                validScreenPoints[i2]
            else {
                continue
            }

            let uv0 = textureCoordinates[i0]
            let uv1 = textureCoordinates[i1]
            let uv2 = textureCoordinates[i2]
            guard
                uv0.x.isFinite,
                uv0.y.isFinite,
                uv1.x.isFinite,
                uv1.y.isFinite,
                uv2.x.isFinite,
                uv2.y.isFinite
            else {
                continue
            }

            let area = abs(
                (screenPoints[i0].x * (screenPoints[i1].y - screenPoints[i2].y))
                    + (screenPoints[i1].x * (screenPoints[i2].y - screenPoints[i0].y))
                    + (screenPoints[i2].x * (screenPoints[i0].y - screenPoints[i1].y))
            ) * 0.5
            guard area > 0.04 else { continue }

            triangles.append(
                ARKitFaceUVProjectionMap.Triangle(
                    uv0: CGPoint(x: CGFloat(uv0.x), y: CGFloat(uv0.y)),
                    uv1: CGPoint(x: CGFloat(uv1.x), y: CGFloat(uv1.y)),
                    uv2: CGPoint(x: CGFloat(uv2.x), y: CGFloat(uv2.y)),
                    screen0: screenPoints[i0],
                    screen1: screenPoints[i1],
                    screen2: screenPoints[i2]
                )
            )
        }

        guard triangles.count > 80 else { return nil }
        return ARKitFaceUVProjectionMap(triangles: triangles)
    }

    private func hideStaleLipParsingOverlay() {
        let now = CACurrentMediaTime()
        if let latestLipParsingUpdatedAt, now - latestLipParsingUpdatedAt < 1.1 {
            return
        }

        resetLipTrackingStateForFaceLoss()
    }

    private func resetLipTrackingStateForFaceLoss() {
        lipParsingGeneration += 1
        isLipParsingInFlight = false
        isRawUVHotStartInFlight = false
        lastLipParsingStartedAt = 0
        latestLipParsingUpdatedAt = nil
        latestLipParsingDurationMS = nil
        latestLipParsingBreakdownText = nil
        latestLipRenderDurationMS = nil
        latestFaceParsingMask = nil
        latestFaceUVMask = nil
        latestScreenLipMask = nil
        latestScreenLipMaskUpdatedAt = nil
        latestScreenLipDisplayFrame = nil
        screenLipMaskTracker.reset()
        latestUVLipAlphaMask = nil
        latestUVLipSmoothingSourceMask = nil
        latestUVLipColorSupportMask = nil
        latestUVLipLineRenderMask = nil
        latestUVLipLineDebugMask = nil
        latestUVLipLineFusedDebugMask = nil
        latestUVLipLineOnlyDebugMask = nil
        latestUVLipLineColorDebugMask = nil
        latestCurveAssistMask = nil
        curveRejectStreak = 0
        curveBackoffFramesRemaining = 0
        uvLipMaskLock.reset()
        uvLipLineFusedMaskLock.reset()
        uvLipLineOnlyMaskLock.reset()
        uvLipLineColorMaskLock.reset()
        activeAutoLipRenderPath = currentLipMode == .autoFusion ? .none : activeAutoLipRenderPath
        textureRenderGeneration += 1
        needsTextureRefresh = true
        material.diffuse.contents = UIColor.clear
        lipParsingOverlayLayer.contents = nil
        lipParsingOverlayLayer.isHidden = true
        clearPreciseLipMakeupLayer()
        updateLipOverlayGeometry(nil)
    }
}

private extension CGRect {
    func interpolated(to rect: CGRect, amount: CGFloat) -> CGRect {
        let clampedAmount = min(max(amount, 0), 1)
        return CGRect(
            x: origin.x + ((rect.origin.x - origin.x) * clampedAmount),
            y: origin.y + ((rect.origin.y - origin.y) * clampedAmount),
            width: size.width + ((rect.size.width - size.width) * clampedAmount),
            height: size.height + ((rect.size.height - size.height) * clampedAmount)
        )
    }
}
