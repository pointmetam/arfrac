//
//  ARFaceMakeupView.swift
//  arface
//
//  Created by lee on 2026/5/21.
//

import AVFoundation
import CoreImage
import CoreML
import SwiftUI
import UIKit
import Vision

#if canImport(MediaPipeTasksVision)
import MediaPipeTasksVision
#endif

private extension Lip2DCameraPosition {
    var captureDevicePosition: AVCaptureDevice.Position {
        switch self {
        case .front:
            .front
        case .back:
            .back
        }
    }

    var visionImageOrientation: CGImagePropertyOrientation {
        switch self {
        case .front:
            .leftMirrored
        case .back:
            .right
        }
    }

    var mediaPipeImageOrientation: UIImage.Orientation {
        switch self {
        case .front:
            .leftMirrored
        case .back:
            .right
        }
    }
}

struct ARFaceMakeupView: UIViewRepresentable {
    @ObservedObject var state: MakeupState

    func makeCoordinator() -> Coordinator {
        Coordinator(state: state, cameraPosition: state.lip2DCameraPosition)
    }

    func makeUIView(context: Context) -> LipstickCameraView {
        let cameraView = LipstickCameraView()
        context.coordinator.attach(to: cameraView)
        context.coordinator.apply(
            look: state.selectedLook,
            intensity: state.intensity,
            makeupVisible: state.isMakeupVisible && !state.isRecordingLip2D,
            showFaceMesh: state.showFaceMesh
        )
        context.coordinator.setLip2DRecordingActive(state.isRecordingLip2D)
        context.coordinator.startIfPossible()
        return cameraView
    }

    func updateUIView(_ cameraView: LipstickCameraView, context: Context) {
        context.coordinator.apply(
            look: state.selectedLook,
            intensity: state.intensity,
            makeupVisible: state.isMakeupVisible && !state.isRecordingLip2D,
            showFaceMesh: state.showFaceMesh
        )
        context.coordinator.setLip2DRecordingActive(state.isRecordingLip2D)
        context.coordinator.startIfPossible()
    }

    static func dismantleUIView(_ cameraView: LipstickCameraView, coordinator: Coordinator) {
        coordinator.setLip2DRecordingActive(false)
        coordinator.stop()
    }
}

final class LipstickCameraView: UIView {
    let lipOverlay = LipMakeupOverlayView()

    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        previewLayer.videoGravity = .resizeAspectFill
        installOverlay()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func installOverlay() {
        lipOverlay.translatesAutoresizingMaskIntoConstraints = false
        lipOverlay.isUserInteractionEnabled = false
        addSubview(lipOverlay)
        NSLayoutConstraint.activate([
            lipOverlay.leadingAnchor.constraint(equalTo: leadingAnchor),
            lipOverlay.trailingAnchor.constraint(equalTo: trailingAnchor),
            lipOverlay.topAnchor.constraint(equalTo: topAnchor),
            lipOverlay.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }
}

final class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private weak var cameraView: LipstickCameraView?
    private let state: MakeupState
    private let cameraPosition: Lip2DCameraPosition
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.perfectmakeup.lipstick.session")
    private let videoQueue = DispatchQueue(label: "com.perfectmakeup.lipstick.video", qos: .userInteractive)
    private let detectionLock = NSLock()
    private let lip2DRecordingLock = NSLock()
    private let coreMLParsingQueue = DispatchQueue(label: "com.perfectmakeup.lipstick.coreml", qos: .userInitiated)
    private let coreMLParsingLock = NSLock()
    private let lipParsingModel = LipParsingModel()
    private let lip2DTrainingRecorder = Lip2DTrainingSampleRecorder()

    private var sessionConfigured = false
    private var sessionStarted = false
    private var detectionInFlight = false
    private var lip2DRecordingActive = false
    private var coreMLParsingInFlight = false
    private var latestCoreMLParsingMask: LipPixelMask?
    private var latestCoreMLParsingUpdatedAt: CFTimeInterval?
    private var latestCoreMLDurationMS: Double?
    private var performanceWindowStart = CACurrentMediaTime()
    private var performanceWindowFrames = 0
    private var renderedFPS: Double = 0
    private var lastPerformancePublishAt: CFTimeInterval = 0
    private var lastPerformanceText = ""
    private var appliedLook: MakeupLook?
    private var appliedIntensity: Double = -1
    private var appliedMakeupVisibility = true
    private var appliedMeshVisibility = false
    private var latestVisionShape: LipLandmarkShape?
    private var visionAnchorMissCount = 0
    private var mediaPipeFrameCount = 0
    private var coreMLParsingFrameCount = 0
    private let colorFrameLock = NSLock()
    private var pendingColorFrames: [Int: LipColorFrame] = [:]

    #if canImport(MediaPipeTasksVision)
    private var faceLandmarker: FaceLandmarker?
    private var mediaPipeFailureMessage: String?
    private var latestMediaPipeTimestamp = 0
    private var mediaPipeProjection: LipLandmarkShape.MediaPipeProjection?
    #endif

    private static let usesCoreMLParsingMask = false
    private static let coreMLParsingFrameInterval = 6
    private static let coreMLParsingMaskTTL: CFTimeInterval = 0.24

    init(state: MakeupState, cameraPosition: Lip2DCameraPosition) {
        self.state = state
        self.cameraPosition = cameraPosition
        super.init()
        lip2DTrainingRecorder.cameraPosition = cameraPosition
        lip2DTrainingRecorder.onStatusChange = { [weak state] text in
            Task { @MainActor in
                state?.recordingText = text
            }
        }
        lip2DTrainingRecorder.onRecordingActiveChange = { [weak state] active in
            Task { @MainActor in
                guard state?.isRecordingLip2D != active else { return }
                state?.isRecordingLip2D = active
            }
        }
    }

    func attach(to cameraView: LipstickCameraView) {
        self.cameraView = cameraView
        cameraView.previewLayer.session = captureSession
        configurePreviewConnection()
    }

    func startIfPossible() {
        guard !sessionStarted else { return }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStartSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    granted ? self?.configureAndStartSession() : self?.updateStatus(.failed("相机权限未开启"))
                }
            }
        case .denied, .restricted:
            updateStatus(.failed("相机权限未开启"))
        @unknown default:
            updateStatus(.failed("无法访问相机"))
        }
    }

    func stop() {
        setLip2DRecordingActive(false)
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
            self.sessionStarted = false
        }
    }

    func setLip2DRecordingActive(_ active: Bool) {
        let changed = updateLip2DRecordingFlag(active)
        guard changed else { return }

        if active {
            endDetection()
            cameraView?.lipOverlay.update(shape: nil)
            updatePerformanceText("")
            lip2DTrainingRecorder.start()
        } else {
            lip2DTrainingRecorder.stop()
        }
    }

    private func updateLip2DRecordingFlag(_ active: Bool) -> Bool {
        lip2DRecordingLock.lock()
        defer { lip2DRecordingLock.unlock() }

        guard lip2DRecordingActive != active else { return false }
        lip2DRecordingActive = active
        return true
    }

    private func isLip2DRecordingActive() -> Bool {
        lip2DRecordingLock.lock()
        defer { lip2DRecordingLock.unlock() }

        return lip2DRecordingActive
    }

    func apply(look: MakeupLook, intensity: Double, makeupVisible: Bool, showFaceMesh: Bool) {
        let intensityChanged = abs(appliedIntensity - intensity) > 0.01
        guard
            appliedLook != look ||
            intensityChanged ||
            appliedMakeupVisibility != makeupVisible ||
            appliedMeshVisibility != showFaceMesh
        else {
            return
        }

        appliedLook = look
        appliedIntensity = intensity
        appliedMakeupVisibility = makeupVisible
        appliedMeshVisibility = showFaceMesh

        cameraView?.lipOverlay.updateStyle(
            color: look.lip.uiColor(intensity: makeupVisible ? intensity : 0),
            showsDebugLandmarks: showFaceMesh
        )

        if showFaceMesh {
            publishPerformance(force: true)
        } else {
            updatePerformanceText("")
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        if isLip2DRecordingActive() {
            lip2DTrainingRecorder.record(sampleBuffer: sampleBuffer)
            return
        }

        guard beginDetection() else { return }
        recordProcessedFrame()

        #if canImport(MediaPipeTasksVision)
        if let faceLandmarker {
            detectWithMediaPipe(sampleBuffer: sampleBuffer, faceLandmarker: faceLandmarker)
        } else {
            detectWithVision(sampleBuffer: sampleBuffer)
        }
        #else
        detectWithVision(sampleBuffer: sampleBuffer)
        #endif
    }

    private func configureAndStartSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }

            do {
                try self.configureSessionIfNeeded()
            } catch {
                self.updateStatus(.failed(error.localizedDescription))
                return
            }

            guard !self.captureSession.isRunning else { return }
            self.captureSession.startRunning()
            self.sessionStarted = true

            DispatchQueue.main.async { [weak self] in
                self?.configurePreviewConnection()
                self?.updateStatus(.ready)
            }
        }
    }

    private func configureSessionIfNeeded() throws {
        guard !sessionConfigured else { return }

        captureSession.beginConfiguration()
        if captureSession.canSetSessionPreset(.hd1920x1080) {
            captureSession.sessionPreset = .hd1920x1080
        } else if captureSession.canSetSessionPreset(.high) {
            captureSession.sessionPreset = .high
        } else {
            captureSession.sessionPreset = .medium
        }
        defer { captureSession.commitConfiguration() }

        guard let camera = Self.camera(position: cameraPosition) else {
            throw LipstickCameraError.cameraUnavailable(cameraPosition)
        }

        let input = try AVCaptureDeviceInput(device: camera)
        guard captureSession.canAddInput(input) else {
            throw LipstickCameraError.cannotAddCameraInput(cameraPosition)
        }
        captureSession.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        guard captureSession.canAddOutput(videoOutput) else {
            throw LipstickCameraError.cannotAddVideoOutput
        }
        captureSession.addOutput(videoOutput)
        configureVideoConnection()

        #if canImport(MediaPipeTasksVision)
        if !isLip2DRecordingActive() {
            configureFaceLandmarkerIfPossible()
        }
        #endif

        sessionConfigured = true
    }

    private func configureVideoConnection() {
        guard let connection = videoOutput.connection(with: .video) else { return }

        if connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }

        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
    }

    private func configurePreviewConnection() {
        guard let connection = cameraView?.previewLayer.connection else { return }

        if connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }

        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = cameraPosition.previewMirrored
        }
    }

    private func detectWithVision(sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            endDetection()
            return
        }

        let request = VNDetectFaceLandmarksRequest { [weak self] request, _ in
            guard let self else { return }
            let observation = (request.results as? [VNFaceObservation])?.first

            DispatchQueue.main.async { [weak self] in
                guard let self, let previewLayer = self.cameraView?.previewLayer else {
                    self?.endDetection()
                    return
                }
                guard !self.isLip2DRecordingActive() else {
                    self.endDetection()
                    return
                }

                let shape = observation.flatMap {
                    LipLandmarkShape(visionObservation: $0, previewLayer: previewLayer)
                }
                let colorFrame = LipColorFrame(pixelBuffer: pixelBuffer)
                let displayShape = shape.map { self.displayShape(for: $0, bounds: previewLayer.bounds) }
                let overlayShape = shape.map { self.overlayInputShape(for: $0, bounds: previewLayer.bounds) }
                let segmentation = displayShape.flatMap {
                    colorFrame.segmentation(
                        for: $0,
                        previewLayer: previewLayer
                    )
                }
                let parsingMask = self.cachedCoreMLParsingMask()
                self.scheduleCoreMLParsingIfNeeded(
                    colorFrame: colorFrame,
                    displayShape: displayShape,
                    previewLayer: previewLayer
                )
                self.cameraView?.lipOverlay.update(
                    shape: overlayShape,
                    segmentedDisplayShape: segmentation?.shape,
                    colorRefinement: segmentation?.refinement,
                    segmentationConfidence: segmentation?.confidence,
                    segmentationMask: parsingMask ?? segmentation?.mask
                )
                self.updateStatus(shape == nil ? .limited("请正对镜头") : .tracking)
                self.endDetection()
            }
        }

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: cameraPosition.visionImageOrientation,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.updateStatus(.limited("嘴唇识别不稳定"))
                self?.endDetection()
            }
        }
    }

    #if canImport(MediaPipeTasksVision)
    private func configureFaceLandmarkerIfPossible() {
        guard faceLandmarker == nil else { return }

        guard let modelPath = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else {
            mediaPipeFailureMessage = "缺少 face_landmarker.task，暂用 Vision"
            return
        }

        do {
            let baseOptions = BaseOptions()
            baseOptions.modelAssetPath = modelPath

            let options = FaceLandmarkerOptions()
            options.baseOptions = baseOptions
            options.runningMode = .liveStream
            options.numFaces = 1
            options.minFaceDetectionConfidence = 0.34
            options.minFacePresenceConfidence = 0.34
            options.minTrackingConfidence = 0.24
            options.outputFaceBlendshapes = false
            options.outputFacialTransformationMatrixes = false
            options.faceLandmarkerLiveStreamDelegate = self

            faceLandmarker = try FaceLandmarker(options: options)
        } catch {
            mediaPipeFailureMessage = "MediaPipe 初始化失败，暂用 Vision"
        }
    }

    private func detectWithMediaPipe(sampleBuffer: CMSampleBuffer, faceLandmarker: FaceLandmarker) {
        do {
            mediaPipeFrameCount += 1
            if latestVisionShape == nil || mediaPipeProjection == nil || mediaPipeFrameCount % 45 == 0 {
                refreshVisionAnchor(sampleBuffer: sampleBuffer)
            }

            let image = try MPImage(sampleBuffer: sampleBuffer, orientation: cameraPosition.mediaPipeImageOrientation)
            let timestamp = monotonicTimestamp(from: sampleBuffer)
            storeColorFrame(sampleBuffer: sampleBuffer, timestamp: timestamp)
            try faceLandmarker.detectAsync(
                image: image,
                timestampInMilliseconds: timestamp
            )
        } catch {
            endDetection()
            detectWithVision(sampleBuffer: sampleBuffer)
        }
    }

    private func monotonicTimestamp(from sampleBuffer: CMSampleBuffer) -> Int {
        let seconds = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let proposedTimestamp = seconds.isFinite ? Int(seconds * 1000) : latestMediaPipeTimestamp + 1
        latestMediaPipeTimestamp = max(proposedTimestamp, latestMediaPipeTimestamp + 1)
        return latestMediaPipeTimestamp
    }

    private func refreshVisionAnchor(sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNDetectFaceLandmarksRequest { [weak self] request, _ in
            let observation = (request.results as? [VNFaceObservation])?.first

            DispatchQueue.main.async { [weak self] in
                guard let self, let previewLayer = self.cameraView?.previewLayer else { return }
                if let shape = observation.flatMap({
                    LipLandmarkShape(visionObservation: $0, previewLayer: previewLayer)
                }) {
                    self.latestVisionShape = shape
                    self.visionAnchorMissCount = 0
                } else {
                    self.visionAnchorMissCount += 1
                    if self.visionAnchorMissCount > 18 {
                        self.latestVisionShape = nil
                    }
                }
            }
        }

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: cameraPosition.visionImageOrientation,
            options: [:]
        )

        try? handler.perform([request])
    }

    private func storeColorFrame(sampleBuffer: CMSampleBuffer, timestamp: Int) {
        guard let frame = LipColorFrame(sampleBuffer: sampleBuffer) else { return }

        colorFrameLock.lock()
        pendingColorFrames[timestamp] = frame

        if pendingColorFrames.count > 4 {
            let staleKeys = pendingColorFrames.keys.sorted().dropLast(4)
            for key in staleKeys {
                pendingColorFrames.removeValue(forKey: key)
            }
        }

        colorFrameLock.unlock()
    }

    private func takeColorFrame(timestamp: Int) -> LipColorFrame? {
        colorFrameLock.lock()
        let frame = pendingColorFrames.removeValue(forKey: timestamp)
        colorFrameLock.unlock()
        return frame
    }
    #endif

    private func beginDetection() -> Bool {
        detectionLock.lock()
        defer { detectionLock.unlock() }

        guard !detectionInFlight else { return false }
        detectionInFlight = true
        return true
    }

    private func endDetection() {
        detectionLock.lock()
        detectionInFlight = false
        detectionLock.unlock()
    }

    private func recordProcessedFrame() {
        let now = CACurrentMediaTime()
        performanceWindowFrames += 1
        let elapsed = now - performanceWindowStart
        guard elapsed >= 0.50 else { return }

        renderedFPS = Double(performanceWindowFrames) / elapsed
        performanceWindowFrames = 0
        performanceWindowStart = now
        publishPerformance(now: now)
    }

    private func shouldRunCoreMLParsing() -> Bool {
        guard Self.usesCoreMLParsingMask else { return false }
        coreMLParsingFrameCount += 1
        return coreMLParsingFrameCount == 1
            || coreMLParsingFrameCount % Self.coreMLParsingFrameInterval == 0
    }

    private func cachedCoreMLParsingMask() -> LipPixelMask? {
        guard Self.usesCoreMLParsingMask else { return nil }

        coreMLParsingLock.lock()
        defer { coreMLParsingLock.unlock() }

        guard
            let mask = latestCoreMLParsingMask,
            let updatedAt = latestCoreMLParsingUpdatedAt,
            CACurrentMediaTime() - updatedAt < Self.coreMLParsingMaskTTL
        else {
            return nil
        }

        return mask
    }

    private func scheduleCoreMLParsingIfNeeded(
        colorFrame: LipColorFrame?,
        displayShape: LipLandmarkShape?,
        previewLayer: AVCaptureVideoPreviewLayer
    ) {
        guard
            Self.usesCoreMLParsingMask,
            shouldRunCoreMLParsing(),
            let colorFrame,
            let displayShape,
            let request = colorFrame.parsingRequest(
                for: displayShape,
                previewLayer: previewLayer,
                model: lipParsingModel
            )
        else {
            return
        }

        coreMLParsingLock.lock()
        guard !coreMLParsingInFlight else {
            coreMLParsingLock.unlock()
            return
        }
        coreMLParsingInFlight = true
        coreMLParsingLock.unlock()

        let model = lipParsingModel
        let startedAt = CACurrentMediaTime()
        coreMLParsingQueue.async { [weak self] in
            let mask = model.mask(request: request)
            let durationMS = (CACurrentMediaTime() - startedAt) * 1000

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.coreMLParsingLock.lock()
                self.coreMLParsingInFlight = false
                self.latestCoreMLDurationMS = durationMS
                if let mask {
                    self.latestCoreMLParsingMask = mask
                    self.latestCoreMLParsingUpdatedAt = CACurrentMediaTime()
                }
                self.coreMLParsingLock.unlock()
                self.publishPerformance(force: true)
            }
        }
    }

    private func publishPerformance(now: CFTimeInterval = CACurrentMediaTime(), force: Bool = false) {
        guard appliedMeshVisibility else { return }
        guard force || now - lastPerformancePublishAt > 0.35 else { return }

        coreMLParsingLock.lock()
        let isInFlight = coreMLParsingInFlight
        let duration = latestCoreMLDurationMS
        let maskAge = latestCoreMLParsingUpdatedAt.map { max((now - $0) * 1000, 0) }
        coreMLParsingLock.unlock()

        let fpsText = String(format: "FPS %.0f", renderedFPS)
        let coreMLText = duration.map { String(format: "ML %.0fms", $0) } ?? "ML --"
        let maskText = maskAge.map { String(format: "Mask %.0fms", $0) } ?? "Mask --"
        let inFlightText = isInFlight ? " busy" : ""
        let text = "\(fpsText)  \(coreMLText)  \(maskText)\(inFlightText)"

        lastPerformancePublishAt = now
        updatePerformanceText(text)
    }

    private func updatePerformanceText(_ text: String) {
        guard text != lastPerformanceText else { return }
        lastPerformanceText = text
        DispatchQueue.main.async { [weak self] in
            self?.state.performanceText = text
        }
    }

    private func updateStatus(_ status: MakeupTrackingStatus) {
        DispatchQueue.main.async { [weak self] in
            guard self?.state.status != status else { return }
            self?.state.status = status
        }
    }

    private func displayShape(for shape: LipLandmarkShape, bounds: CGRect) -> LipLandmarkShape {
        cameraPosition.previewMirrored ? shape.horizontallyMirrored(in: bounds) : shape
    }

    private func overlayInputShape(for shape: LipLandmarkShape, bounds: CGRect) -> LipLandmarkShape {
        cameraPosition.previewMirrored ? shape : shape.horizontallyMirrored(in: bounds)
    }

    private static func camera(position: Lip2DCameraPosition) -> AVCaptureDevice? {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: cameraDeviceTypes(position: position),
            mediaType: .video,
            position: position.captureDevicePosition
        )
        return discoverySession.devices.first
    }

    private static func cameraDeviceTypes(position: Lip2DCameraPosition) -> [AVCaptureDevice.DeviceType] {
        switch position {
        case .front:
            [.builtInTrueDepthCamera, .builtInWideAngleCamera]
        case .back:
            [.builtInWideAngleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInTripleCamera]
        }
    }
}

#if canImport(MediaPipeTasksVision)
extension Coordinator: FaceLandmarkerLiveStreamDelegate {
    func faceLandmarker(
        _ faceLandmarker: FaceLandmarker,
        didFinishDetection result: FaceLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        defer { endDetection() }

        guard !isLip2DRecordingActive() else { return }

        guard error == nil, let landmarks = result?.faceLandmarks.first else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                let fallbackShape = self.latestVisionShape
                let overlayShape = fallbackShape.map {
                    self.overlayInputShape(for: $0, bounds: self.cameraView?.previewLayer.bounds ?? .zero)
                }
                self.cameraView?.lipOverlay.update(shape: overlayShape)
                self.updateStatus(fallbackShape == nil ? .limited("请正对镜头") : .tracking)
            }
            return
        }

        let colorFrame = takeColorFrame(timestamp: timestampInMilliseconds)
        DispatchQueue.main.async { [weak self] in
            guard let self, let cameraView = self.cameraView else { return }
            let anchorShape = self.latestVisionShape
            let mediaPipeShape = LipLandmarkShape(
                mediaPipeLandmarks: landmarks,
                previewLayer: cameraView.previewLayer,
                anchorShape: anchorShape,
                preferredProjection: self.mediaPipeProjection
            )
            let shape = mediaPipeShape ?? (self.mediaPipeProjection == nil ? anchorShape : nil)
            if let projection = shape?.mediaPipeProjection {
                self.mediaPipeProjection = projection
            }
            let displayShape = shape.map { self.displayShape(for: $0, bounds: cameraView.previewLayer.bounds) }
            let overlayShape = shape.map { self.overlayInputShape(for: $0, bounds: cameraView.previewLayer.bounds) }
            let segmentation = displayShape.flatMap {
                colorFrame?.segmentation(
                    for: $0,
                    previewLayer: cameraView.previewLayer
                )
            }
            let parsingMask = self.cachedCoreMLParsingMask()
            self.scheduleCoreMLParsingIfNeeded(
                colorFrame: colorFrame,
                displayShape: displayShape,
                previewLayer: cameraView.previewLayer
            )
            cameraView.lipOverlay.update(
                shape: overlayShape,
                segmentedDisplayShape: segmentation?.shape,
                colorRefinement: segmentation?.refinement,
                segmentationConfidence: segmentation?.confidence,
                segmentationMask: parsingMask ?? segmentation?.mask
            )
            self.updateStatus(shape == nil ? .limited("请正对镜头") : .tracking)
        }
    }
}
#endif

struct LipColorRefinement {
    let upperCoverageY: CGFloat
    let upperFillY: CGFloat
    let upperFeatherY: CGFloat
    let cornerCoverageBoost: CGFloat
    let cornerFillBoost: CGFloat
    let cornerFeatherBoost: CGFloat

    static let fallback = LipColorRefinement(
        upperCoverageY: 1.02,
        upperFillY: 0.99,
        upperFeatherY: 1.03,
        cornerCoverageBoost: 0.04,
        cornerFillBoost: 0.03,
        cornerFeatherBoost: 0.04
    )

    func smoothed(toward refinement: LipColorRefinement, amount: CGFloat) -> LipColorRefinement {
        LipColorRefinement(
            upperCoverageY: Self.interpolate(upperCoverageY, refinement.upperCoverageY, amount: amount),
            upperFillY: Self.interpolate(upperFillY, refinement.upperFillY, amount: amount),
            upperFeatherY: Self.interpolate(upperFeatherY, refinement.upperFeatherY, amount: amount),
            cornerCoverageBoost: Self.interpolate(cornerCoverageBoost, refinement.cornerCoverageBoost, amount: amount),
            cornerFillBoost: Self.interpolate(cornerFillBoost, refinement.cornerFillBoost, amount: amount),
            cornerFeatherBoost: Self.interpolate(cornerFeatherBoost, refinement.cornerFeatherBoost, amount: amount)
        )
    }

    private static func interpolate(_ current: CGFloat, _ next: CGFloat, amount: CGFloat) -> CGFloat {
        current + (next - current) * amount
    }
}

struct LipSegmentationResult {
    let shape: LipLandmarkShape
    let refinement: LipColorRefinement
    let confidence: CGFloat
    let mask: LipPixelMask?
}

struct LipPixelMask {
    let frame: CGRect
    let width: Int
    let height: Int
    let alpha: [UInt8]

    struct RenderGuide {
        let outerPath: UIBezierPath
        let innerMouthPath: UIBezierPath?
        let upperBoundary: [CGPoint]?
        let lowerBoundary: [CGPoint]?
        let lipBounds: CGRect
    }

    func refined(with guide: RenderGuide) -> LipPixelMask {
        guard width > 0, height > 0, alpha.count == width * height else { return self }

        var refinedAlpha = alpha
        let sortedUpperBoundary = guide.upperBoundary?.sorted { $0.x < $1.x }
        let sortedLowerBoundary = guide.lowerBoundary?.sorted { $0.x < $1.x }
        let upperFeather = max(guide.lipBounds.height * 0.040, 1.75)
        let lowerFeather = max(guide.lipBounds.height * 0.075, 2.75)
        let cornerFeather = max(guide.lipBounds.width * 0.095, 5.0)
        let edgeFeatherRadius = min(max(Int((CGFloat(width) / max(frame.width, 1) * 4.2).rounded()), 3), 7)

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard refinedAlpha[index] > 0 else { continue }

                let point = CGPoint(
                    x: frame.minX + (CGFloat(x) + 0.5) * frame.width / CGFloat(width),
                    y: frame.minY + (CGFloat(y) + 0.5) * frame.height / CGFloat(height)
                )
                if
                    let sortedUpperBoundary,
                    let upperY = Self.interpolatedY(on: sortedUpperBoundary, at: point.x),
                    point.y < upperY
                {
                    refinedAlpha[index] = 0
                    continue
                }

                if !guide.outerPath.contains(point) {
                    refinedAlpha[index] = 0
                    continue
                }

                if guide.innerMouthPath?.contains(point) == true {
                    refinedAlpha[index] = 0
                    continue
                }

                var opacity = CGFloat(refinedAlpha[index]) / 255
                if
                    let sortedUpperBoundary,
                    let upperY = Self.interpolatedY(on: sortedUpperBoundary, at: point.x)
                {
                    let distance = point.y - upperY
                    opacity *= Self.smoothstep(distance / upperFeather)
                }

                if
                    let sortedLowerBoundary,
                    let lowerY = Self.interpolatedY(on: sortedLowerBoundary, at: point.x)
                {
                    let distance = lowerY - point.y
                    if distance < 0 {
                        refinedAlpha[index] = 0
                        continue
                    }

                    opacity *= Self.smoothstep(distance / lowerFeather)
                }

                let leftProgress = (point.x - guide.lipBounds.minX) / max(cornerFeather, 0.001)
                let rightProgress = (guide.lipBounds.maxX - point.x) / max(cornerFeather, 0.001)
                let cornerFade = min(Self.smoothstep(leftProgress), Self.smoothstep(rightProgress))
                if cornerFade < 1 {
                    opacity *= 0.46 + cornerFade * 0.54
                }

                refinedAlpha[index] = UInt8(min(max(opacity * 255, 0), 255))
            }
        }

        refinedAlpha = Self.edgeFeathered(
            alpha: refinedAlpha,
            width: width,
            height: height,
            radius: edgeFeatherRadius
        )

        return LipPixelMask(frame: frame, width: width, height: height, alpha: refinedAlpha)
    }

    private static func interpolatedY(on points: [CGPoint], at x: CGFloat) -> CGFloat? {
        guard points.count >= 2 else { return nil }
        guard let first = points.first, let last = points.last, x >= first.x, x <= last.x else {
            return nil
        }

        for index in 1..<points.count {
            let previous = points[index - 1]
            let current = points[index]
            let minX = min(previous.x, current.x)
            let maxX = max(previous.x, current.x)
            guard x >= minX, x <= maxX else { continue }

            let progress = (x - previous.x) / max(current.x - previous.x, 0.001)
            return previous.y + (current.y - previous.y) * progress
        }

        return nil
    }

    private static func smoothstep(_ value: CGFloat) -> CGFloat {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }

    private static func edgeFeathered(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 2, height > 2, alpha.count == width * height else {
            return alpha
        }

        var output = alpha
        let radiusSquared = radius * radius

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard alpha[index] > 0 else { continue }

                var nearestZeroSquared = radiusSquared + 1
                let minY = max(0, y - radius)
                let maxY = min(height - 1, y + radius)
                let minX = max(0, x - radius)
                let maxX = min(width - 1, x + radius)

                for sampleY in minY...maxY {
                    for sampleX in minX...maxX {
                        let dy = sampleY - y
                        let dx = sampleX - x
                        let distanceSquared = dx * dx + dy * dy
                        guard distanceSquared <= radiusSquared else { continue }

                        if alpha[sampleY * width + sampleX] == 0 {
                            nearestZeroSquared = min(nearestZeroSquared, distanceSquared)
                        }
                    }
                }

                guard nearestZeroSquared <= radiusSquared else { continue }

                let distance = sqrt(CGFloat(nearestZeroSquared))
                let feather = smoothstep(distance / CGFloat(radius))
                let minimumEdgeOpacity: CGFloat = 0.18
                let opacityScale = minimumEdgeOpacity + feather * (1 - minimumEdgeOpacity)
                output[index] = UInt8(min(max(CGFloat(alpha[index]) * opacityScale, 0), 255))
            }
        }

        return output
    }

    func tintedImage(color: UIColor) -> CGImage? {
        guard width > 0, height > 0, alpha.count == width * height else { return nil }

        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var colorAlpha: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &colorAlpha)

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for index in alpha.indices {
            let maskAlpha = CGFloat(alpha[index]) / 255
            let combinedAlpha = min(max(maskAlpha * colorAlpha, 0), 1)
            let pixelOffset = index * 4
            pixels[pixelOffset] = UInt8(min(max(red * combinedAlpha * 255, 0), 255))
            pixels[pixelOffset + 1] = UInt8(min(max(green * combinedAlpha * 255, 0), 255))
            pixels[pixelOffset + 2] = UInt8(min(max(blue * combinedAlpha * 255, 0), 255))
            pixels[pixelOffset + 3] = UInt8(min(max(combinedAlpha * 255, 0), 255))
        }

        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

private struct LipParsingRequest: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let roi: CGRect
    let topLeft: CGPoint
    let topRight: CGPoint
    let bottomLeft: CGPoint
    let bottomRight: CGPoint
}

private final class LipParsingModel {
    private static let inputSize = 256
    private static let classCount = 9
    private static let upperLipClass = 5
    private static let lowerLipClass = 6
    private static let innerMouthClass = 7

    private let model: MLModel?
    private let inputName: String?
    private let outputName: String?

    init() {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let modelURL = try Self.loadableModelURL()
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            self.model = model
            self.inputName = model.modelDescription.inputDescriptionsByName.keys.first
            self.outputName = model.modelDescription.outputDescriptionsByName.keys.first
        } catch {
            self.model = nil
            self.inputName = nil
            self.outputName = nil
        }
    }

    func mask(
        pixelBuffer: CVPixelBuffer,
        displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> LipPixelMask? {
        guard let request = makeRequest(
            pixelBuffer: pixelBuffer,
            displayShape: displayShape,
            previewLayer: previewLayer
        ) else {
            return nil
        }

        return mask(request: request)
    }

    func makeRequest(
        pixelBuffer: CVPixelBuffer,
        displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> LipParsingRequest? {
        guard
            model != nil,
            inputName != nil,
            let roi = Self.faceROI(for: displayShape, in: previewLayer.bounds)
        else {
            return nil
        }

        return LipParsingRequest(
            pixelBuffer: pixelBuffer,
            roi: roi,
            topLeft: previewLayer.captureDevicePointConverted(
                fromLayerPoint: CGPoint(x: roi.minX, y: roi.minY)
            ),
            topRight: previewLayer.captureDevicePointConverted(
                fromLayerPoint: CGPoint(x: roi.maxX, y: roi.minY)
            ),
            bottomLeft: previewLayer.captureDevicePointConverted(
                fromLayerPoint: CGPoint(x: roi.minX, y: roi.maxY)
            ),
            bottomRight: previewLayer.captureDevicePointConverted(
                fromLayerPoint: CGPoint(x: roi.maxX, y: roi.maxY)
            )
        )
    }

    func mask(request: LipParsingRequest) -> LipPixelMask? {
        guard
            let model,
            let inputName
        else {
            return nil
        }

        let input: MLMultiArray
        do {
            input = try makeInputArray(request: request)
        } catch {
            return nil
        }

        do {
            let provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(multiArray: input)]
            )
            let prediction = try model.prediction(from: provider)
            guard
                let outputName = outputName ?? prediction.featureNames.first,
                let logits = prediction.featureValue(for: outputName)?.multiArrayValue
            else {
                return nil
            }

            return makeMask(from: logits, frame: request.roi)
        } catch {
            return nil
        }
    }

    private static func loadableModelURL() throws -> URL {
        let compiledURLs = [
            Bundle.main.url(forResource: "MakeupParsingSmall", withExtension: "mlmodelc"),
            Bundle.main.url(forResource: "MakeupParsingSmall", withExtension: "mlmodelc", subdirectory: "Models")
        ].compactMap { $0 }

        if let compiledURL = compiledURLs.first {
            return compiledURL
        }

        let packageURLs = [
            Bundle.main.url(forResource: "MakeupParsingSmall", withExtension: "mlpackage"),
            Bundle.main.url(forResource: "MakeupParsingSmall", withExtension: "mlpackage", subdirectory: "Models")
        ].compactMap { $0 }

        if let packageURL = packageURLs.first {
            return try MLModel.compileModel(at: packageURL)
        }

        throw CocoaError(.fileNoSuchFile)
    }

    fileprivate static func faceROI(for shape: LipLandmarkShape, in bounds: CGRect) -> CGRect? {
        guard bounds.width > 10, bounds.height > 10 else { return nil }

        let lipBounds = shape.outerBounds
        guard lipBounds.width > 20, lipBounds.height > 6 else { return nil }

        let rawSide = max(lipBounds.width * 4.25, lipBounds.height * 8.4, 128)
        let side = min(rawSide, bounds.width * 0.98, bounds.height * 0.78)
        guard side > 32 else { return nil }

        var originX = lipBounds.midX - side * 0.5
        var originY = lipBounds.midY - side * 0.68
        originX = min(max(originX, bounds.minX), bounds.maxX - side)
        originY = min(max(originY, bounds.minY), bounds.maxY - side)

        return CGRect(x: originX, y: originY, width: side, height: side)
            .intersection(bounds)
    }

    private func makeInputArray(request: LipParsingRequest) throws -> MLMultiArray {
        let input = try MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: 3),
                NSNumber(value: Self.inputSize),
                NSNumber(value: Self.inputSize)
            ],
            dataType: .float32
        )
        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let strides = input.strides.map { $0.intValue }
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        CVPixelBufferLockBaseAddress(request.pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(request.pixelBuffer, .readOnly) }

        guard let sampler = LipPixelSampler(pixelBuffer: request.pixelBuffer), strides.count == 4 else {
            throw CocoaError(.coderInvalidValue)
        }

        for y in 0..<Self.inputSize {
            let verticalProgress = (CGFloat(y) + 0.5) / CGFloat(Self.inputSize)
            for x in 0..<Self.inputSize {
                let horizontalProgress = (CGFloat(x) + 0.5) / CGFloat(Self.inputSize)
                let capturePoint = Self.bilinearPoint(
                    horizontalProgress: horizontalProgress,
                    verticalProgress: verticalProgress,
                    topLeft: request.topLeft,
                    topRight: request.topRight,
                    bottomLeft: request.bottomLeft,
                    bottomRight: request.bottomRight
                )
                let color = sampler.sample(normalizedX: capturePoint.x, normalizedY: capturePoint.y)

                let values: [Float]
                if let color {
                    values = [Float(color.red), Float(color.green), Float(color.blue)]
                } else {
                    values = [0, 0, 0]
                }

                for channel in 0..<3 {
                    let offset = channel * strides[1] + y * strides[2] + x * strides[3]
                    pointer[offset] = (values[channel] - mean[channel]) / std[channel]
                }
            }
        }

        return input
    }

    private static func bilinearPoint(
        horizontalProgress: CGFloat,
        verticalProgress: CGFloat,
        topLeft: CGPoint,
        topRight: CGPoint,
        bottomLeft: CGPoint,
        bottomRight: CGPoint
    ) -> CGPoint {
        let top = topLeft.interpolated(to: topRight, amount: horizontalProgress)
        let bottom = bottomLeft.interpolated(to: bottomRight, amount: horizontalProgress)
        return top.interpolated(to: bottom, amount: verticalProgress)
    }

    private func makeMask(from logits: MLMultiArray, frame: CGRect) -> LipPixelMask? {
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 4, shape[1] >= Self.classCount else { return nil }

        let height = shape[2]
        let width = shape[3]
        guard width > 0, height > 0 else { return nil }

        let strides = logits.strides.map { $0.intValue }
        guard strides.count == 4 else { return nil }

        var alpha = [UInt8](repeating: 0, count: width * height)
        var lipPixelCount = 0

        for y in 0..<height {
            for x in 0..<width {
                let upperScore = value(in: logits, strides: strides, channel: Self.upperLipClass, y: y, x: x)
                let lowerScore = value(in: logits, strides: strides, channel: Self.lowerLipClass, y: y, x: x)
                let mouthScore = value(in: logits, strides: strides, channel: Self.innerMouthClass, y: y, x: x)
                let lipScore = max(upperScore, lowerScore)
                let isUpperLip = upperScore >= lowerScore
                var bestOtherScore = mouthScore

                for channel in 0..<Self.classCount where channel != Self.upperLipClass && channel != Self.lowerLipClass {
                    bestOtherScore = max(bestOtherScore, value(in: logits, strides: strides, channel: channel, y: y, x: x))
                }

                let margin = lipScore - bestOtherScore
                let requiredMargin: Float = isUpperLip ? -0.03 : -0.08
                guard lipScore > mouthScore, margin > requiredMargin else { continue }

                let opacity = isUpperLip
                    ? smoothstep((margin + 0.12) / 0.36)
                    : smoothstep((margin + 0.17) / 0.43)
                let minimumOpacity: Float = isUpperLip ? 0.17 : 0.14
                guard opacity > minimumOpacity else { continue }

                alpha[y * width + x] = UInt8(min(max(opacity * 235, 0), 235))
                lipPixelCount += 1
            }
        }

        let coverage = CGFloat(lipPixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.004, coverage < 0.28 else { return nil }

        alpha = softlyExpanded(alpha: alpha, width: width, height: height, radius: 1)
        alpha = blurred(alpha: alpha, width: width, height: height, radius: 1)
        return LipPixelMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    private func value(
        in array: MLMultiArray,
        strides: [Int],
        channel: Int,
        y: Int,
        x: Int
    ) -> Float {
        let offset = channel * strides[1] + y * strides[2] + x * strides[3]

        if array.dataType == .float32 {
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            return pointer[offset]
        }

        return array[
            [
                NSNumber(value: 0),
                NSNumber(value: channel),
                NSNumber(value: y),
                NSNumber(value: x)
            ]
        ].floatValue
    }

    private func smoothstep(_ value: Float) -> Float {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }

    private func softlyExpanded(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        var output = alpha
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] == 0 {
                var neighborMax = 0
                for sampleY in max(0, y - radius)...y {
                    for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                        neighborMax = max(neighborMax, Int(alpha[sampleY * width + sampleX]))
                    }
                }

                if neighborMax > 70 {
                    output[y * width + x] = UInt8(min(max(CGFloat(neighborMax) * 0.52, 0), 150))
                }
            }
        }

        return output
    }

    private func blurred(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        var horizontal = [UInt8](repeating: 0, count: alpha.count)
        var output = [UInt8](repeating: 0, count: alpha.count)

        for y in 0..<height {
            for x in 0..<width {
                var sum = 0
                var count = 0
                for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                    sum += Int(alpha[y * width + sampleX])
                    count += 1
                }
                horizontal[y * width + x] = UInt8(sum / max(count, 1))
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                var sum = 0
                var count = 0
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    sum += Int(horizontal[sampleY * width + x])
                    count += 1
                }
                output[y * width + x] = UInt8(sum / max(count, 1))
            }
        }

        return output
    }
}

private struct LipColorProfile {
    let lipAverage: LipPixelColor
    let skinAverage: LipPixelColor
    let chromaContrast: CGFloat

    var boundaryThreshold: CGFloat {
        max(chromaContrast * 0.10, 0.006)
    }
}

private struct LipBoundaryResult {
    let point: CGPoint
    let confidence: CGFloat
}

private struct LipPointRefinementResult {
    let points: [CGPoint]?
    let confidence: CGFloat
}

fileprivate struct LipSegmentationOffset {
    let outerOffsets: [CGPoint]
    let innerOffsets: [CGPoint]?

    init?(
        from baseShape: LipLandmarkShape,
        to segmentedShape: LipLandmarkShape,
        maxDistance: CGFloat
    ) {
        guard baseShape.outerPoints.count == segmentedShape.outerPoints.count else {
            return nil
        }

        outerOffsets = baseShape.outerPoints.indices.map {
            Self.clampedOffset(
                from: baseShape.outerPoints[$0],
                to: segmentedShape.outerPoints[$0],
                maxDistance: maxDistance
            )
        }

        if
            let baseInner = baseShape.innerPoints,
            let segmentedInner = segmentedShape.innerPoints,
            baseInner.count == segmentedInner.count
        {
            innerOffsets = baseInner.indices.map {
                Self.clampedOffset(
                    from: baseInner[$0],
                    to: segmentedInner[$0],
                    maxDistance: maxDistance * 0.72
                )
            }
        } else {
            innerOffsets = nil
        }
    }

    private init(outerOffsets: [CGPoint], innerOffsets: [CGPoint]?) {
        self.outerOffsets = outerOffsets
        self.innerOffsets = innerOffsets
    }

    func applied(to shape: LipLandmarkShape) -> LipLandmarkShape? {
        guard outerOffsets.count == shape.outerPoints.count else { return nil }

        let outerPoints = shape.outerPoints.indices.map {
            shape.outerPoints[$0].applying(offset: outerOffsets[$0])
        }
        let innerPoints: [CGPoint]?
        if
            let shapeInner = shape.innerPoints,
            let innerOffsets,
            shapeInner.count == innerOffsets.count
        {
            innerPoints = shapeInner.indices.map {
                shapeInner[$0].applying(offset: innerOffsets[$0])
            }
        } else {
            innerPoints = shape.innerPoints
        }

        return shape.replacing(outerPoints: outerPoints, innerPoints: innerPoints)
    }

    func smoothed(toward offset: LipSegmentationOffset, amount: CGFloat) -> LipSegmentationOffset {
        guard outerOffsets.count == offset.outerOffsets.count else { return offset }

        let outerOffsets = outerOffsets.indices.map {
            self.outerOffsets[$0].interpolated(to: offset.outerOffsets[$0], amount: amount)
        }
        let innerOffsets: [CGPoint]?
        if
            let currentInner = self.innerOffsets,
            let nextInner = offset.innerOffsets,
            currentInner.count == nextInner.count
        {
            innerOffsets = currentInner.indices.map {
                currentInner[$0].interpolated(to: nextInner[$0], amount: amount)
            }
        } else {
            innerOffsets = offset.innerOffsets
        }

        return LipSegmentationOffset(outerOffsets: outerOffsets, innerOffsets: innerOffsets)
    }

    func scaled(by amount: CGFloat) -> LipSegmentationOffset {
        LipSegmentationOffset(
            outerOffsets: outerOffsets.map { $0.scaled(by: amount) },
            innerOffsets: innerOffsets?.map { $0.scaled(by: amount) }
        )
    }

    private static func clampedOffset(
        from basePoint: CGPoint,
        to targetPoint: CGPoint,
        maxDistance: CGFloat
    ) -> CGPoint {
        CGPoint(
            x: targetPoint.x - basePoint.x,
            y: targetPoint.y - basePoint.y
        ).limitedMagnitude(maxDistance)
    }
}

private final class LipColorFrame: @unchecked Sendable {
    private let pixelBuffer: CVPixelBuffer

    init?(sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        self.pixelBuffer = pixelBuffer
    }

    init(pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer
    }

    func refinement(
        for displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> LipColorRefinement? {
        guard displayShape.isPlausible else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let sampler = LipPixelSampler(pixelBuffer: pixelBuffer) else {
            return nil
        }

        guard let profile = colorProfile(for: displayShape, previewLayer: previewLayer, sampler: sampler) else {
            return nil
        }

        return refinement(for: displayShape, previewLayer: previewLayer, sampler: sampler, profile: profile)
    }

    func segmentation(
        for displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> LipSegmentationResult? {
        guard displayShape.isPlausible else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard
            let sampler = LipPixelSampler(pixelBuffer: pixelBuffer),
            let profile = colorProfile(for: displayShape, previewLayer: previewLayer, sampler: sampler)
        else {
            return nil
        }

        let refinedOuter = refineOuterPoints(
            for: displayShape,
            previewLayer: previewLayer,
            sampler: sampler,
            profile: profile
        )
        let refinedInner = refineInnerPoints(
            for: displayShape,
            previewLayer: previewLayer,
            sampler: sampler,
            profile: profile
        )
        let refinedShape = displayShape.replacing(
            outerPoints: refinedOuter.points,
            innerPoints: refinedInner.points
        )
        let segmentationConfidence: CGFloat
        if displayShape.mouthOpeningRatio > 0.08 {
            segmentationConfidence = refinedOuter.confidence * 0.78 + refinedInner.confidence * 0.22
        } else {
            segmentationConfidence = refinedOuter.confidence
        }

        guard refinedOuter.confidence > 0.20, segmentationConfidence > 0.20 else {
            return nil
        }

        let refinement = refinement(
            for: refinedShape,
            previewLayer: previewLayer,
            sampler: sampler,
            profile: profile
        ) ?? .fallback
        let mask = Self.usesExperimentalColorPixelMask
            ? lipPixelMask(
                for: refinedShape,
                previewLayer: previewLayer,
                sampler: sampler,
                profile: profile
            )
            : nil

        return LipSegmentationResult(
            shape: refinedShape,
            refinement: refinement,
            confidence: segmentationConfidence,
            mask: mask
        )
    }

    func parsingMask(
        for displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        model: LipParsingModel
    ) -> LipPixelMask? {
        guard displayShape.isPlausible else { return nil }
        return model.mask(pixelBuffer: pixelBuffer, displayShape: displayShape, previewLayer: previewLayer)
    }

    func parsingRequest(
        for displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        model: LipParsingModel
    ) -> LipParsingRequest? {
        guard displayShape.isPlausible else { return nil }
        return model.makeRequest(pixelBuffer: pixelBuffer, displayShape: displayShape, previewLayer: previewLayer)
    }

    private func colorProfile(
        for displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler
    ) -> LipColorProfile? {
        let lipBounds = displayShape.outerBounds
        let center = displayShape.centerPoint
        let upperPoints = displayShape.outerPoints.filter { $0.y < center.y }
        guard upperPoints.count >= 4 else { return nil }

        let lipInteriorPoints = displayShape.outerPoints.flatMap {
            [
                $0.interpolated(to: center, amount: 0.34),
                $0.interpolated(to: center, amount: 0.52)
            ]
        }
        let upperOffset = max(lipBounds.height * 0.11, 3)
        let upperOutsidePoints = upperPoints.map {
            CGPoint(x: $0.x, y: $0.y - upperOffset)
        }

        let lipSamples = sampleCluster(
            points: lipInteriorPoints,
            radius: 1.6,
            previewLayer: previewLayer,
            sampler: sampler
        )
        let skinSamples = sampleCluster(
            points: upperOutsidePoints,
            radius: 2.0,
            previewLayer: previewLayer,
            sampler: sampler
        )

        guard
            let lipAverage = LipPixelColor.robustAverage(lipSamples),
            let skinAverage = LipPixelColor.robustAverage(skinSamples)
        else {
            return nil
        }

        let chromaContrast = lipAverage.chromaDistance(to: skinAverage)
            + abs(lipAverage.normalizedRedness - skinAverage.normalizedRedness) * 0.45
            + abs(lipAverage.saturation - skinAverage.saturation) * 0.15

        return LipColorProfile(
            lipAverage: lipAverage,
            skinAverage: skinAverage,
            chromaContrast: chromaContrast
        )
    }

    private func refinement(
        for displayShape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler,
        profile: LipColorProfile
    ) -> LipColorRefinement? {
        let lipBounds = displayShape.outerBounds
        let center = displayShape.centerPoint
        let upperPoints = displayShape.outerPoints.filter { $0.y < center.y }
        guard upperPoints.count >= 4 else { return nil }

        let upperOffset = max(lipBounds.height * 0.11, 3)
        let upperOutsidePoints = upperPoints.map {
            CGPoint(x: $0.x, y: $0.y - upperOffset)
        }

        let skinSamples = sample(points: upperOutsidePoints, previewLayer: previewLayer, sampler: sampler)

        let upperLipRatio = lipLikeRatio(
            samples: skinSamples,
            lipAverage: profile.lipAverage,
            skinAverage: profile.skinAverage
        )
        let cornerLipRatio = lipLikeRatio(
            samples: cornerProbePoints(for: displayShape).compactMap {
                sample(point: $0, previewLayer: previewLayer, sampler: sampler)
            },
            lipAverage: profile.lipAverage,
            skinAverage: profile.skinAverage
        )

        return LipColorRefinement(
            upperCoverageY: 0.99 + upperLipRatio * 0.07,
            upperFillY: 0.96 + upperLipRatio * 0.05,
            upperFeatherY: 1.00 + upperLipRatio * 0.07,
            cornerCoverageBoost: 0.03 + cornerLipRatio * 0.05,
            cornerFillBoost: 0.02 + cornerLipRatio * 0.04,
            cornerFeatherBoost: 0.03 + cornerLipRatio * 0.04
        )
    }

    private func refineOuterPoints(
        for shape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler,
        profile: LipColorProfile
    ) -> LipPointRefinementResult {
        let center = shape.centerPoint
        let bounds = shape.outerBounds
        let inwardLimit = max(bounds.height * 0.24, 5)
        let outwardLimit = max(bounds.height * 0.32, 7)
        let maxMove = max(bounds.height * 0.22, 5)
        var confidences: [CGFloat] = []

        let refinedPoints = shape.outerPoints.map { point in
            guard let direction = normalizedDirection(from: center, to: point) else {
                return point
            }

            guard let boundary = lipToSkinBoundary(
                around: point,
                direction: direction,
                inwardLimit: inwardLimit,
                outwardLimit: outwardLimit,
                previewLayer: previewLayer,
                sampler: sampler,
                profile: profile
            ) else {
                return point
            }
            confidences.append(boundary.confidence)

            return limitedPoint(
                from: point,
                toward: boundary.point,
                amount: 0.58 + min(boundary.confidence, 1) * 0.16,
                maxDistance: maxMove
            )
        }

        guard confidences.count >= max(6, shape.outerPoints.count / 3) else {
            return LipPointRefinementResult(points: nil, confidence: 0)
        }

        let confidence = min(
            1,
            confidences.reduce(CGFloat(0), +) / CGFloat(confidences.count)
        )
        return LipPointRefinementResult(points: refinedPoints, confidence: confidence)
    }

    private func refineInnerPoints(
        for shape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler,
        profile: LipColorProfile
    ) -> LipPointRefinementResult {
        guard
            let innerPoints = shape.innerPoints,
            innerPoints.count >= 8,
            shape.mouthOpeningRatio > 0.08
        else {
            return LipPointRefinementResult(points: shape.innerPoints, confidence: 0.5)
        }

        let mouthCenter = averagePoint(innerPoints)
        let bounds = shape.outerBounds
        let lipSideLimit = max(bounds.height * 0.14, 4)
        let mouthSideLimit = max(bounds.height * 0.18, 5)
        let maxMove = max(bounds.height * 0.16, 4)
        var confidences: [CGFloat] = []

        let refinedPoints = innerPoints.map { point in
            guard let direction = normalizedDirection(from: mouthCenter, to: point) else {
                return point
            }

            guard let boundary = mouthToLipBoundary(
                around: point,
                direction: direction,
                mouthSideLimit: mouthSideLimit,
                lipSideLimit: lipSideLimit,
                previewLayer: previewLayer,
                sampler: sampler,
                profile: profile
            ) else {
                return point
            }
            confidences.append(boundary.confidence)

            return limitedPoint(
                from: point,
                toward: boundary.point,
                amount: 0.46 + min(boundary.confidence, 1) * 0.14,
                maxDistance: maxMove
            )
        }

        guard confidences.count >= max(3, innerPoints.count / 4) else {
            return LipPointRefinementResult(points: innerPoints, confidence: 0.18)
        }

        let confidence = min(
            1,
            confidences.reduce(CGFloat(0), +) / CGFloat(confidences.count)
        )
        return LipPointRefinementResult(points: refinedPoints, confidence: confidence)
    }

    private static let usesExperimentalColorPixelMask = false

    private func lipPixelMask(
        for shape: LipLandmarkShape,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler,
        profile: LipColorProfile
    ) -> LipPixelMask? {
        let previewBounds = previewLayer.bounds
        guard previewBounds.width > 1, previewBounds.height > 1 else { return nil }

        let maskFrame = shape.outerBounds
            .expandedBy(widthRatio: 0.20, heightRatio: 0.36)
            .intersection(previewBounds)
        guard maskFrame.width > 12, maskFrame.height > 8 else { return nil }

        let scale = min(220 / maskFrame.width, 150 / maskFrame.height, 1.35)
        let maskWidth = min(max(Int((maskFrame.width * scale).rounded()), 56), 220)
        let maskHeight = min(max(Int((maskFrame.height * scale).rounded()), 36), 150)

        let outerPath = UIBezierPath.smoothClosedPath(
            points: shape
                .cosmeticExpanded(x: 1.10, upperY: 1.08, lowerY: 1.12, cornerXBoost: 0.045)
                .cornerSoftenedOuterPoints(roundness: 0.28)
                .chaikinClosed(iterations: 1)
        )
        let corePath = UIBezierPath.smoothClosedPath(
            points: shape
                .cornerSoftenedOuterPoints(roundness: 0.24)
                .chaikinClosed(iterations: 1)
        )
        let mouthPath: UIBezierPath?
        if shape.mouthOpeningRatio > 0.08, let opening = shape.scaledMouthOpening(x: 0.96, y: 0.98) {
            mouthPath = UIBezierPath.smoothClosedPath(points: opening.chaikinClosed(iterations: 1))
        } else {
            mouthPath = nil
        }

        let threshold = profile.boundaryThreshold
        let denominator = max(profile.chromaContrast * 0.78, 0.014)
        var alpha = [UInt8](repeating: 0, count: maskWidth * maskHeight)

        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let point = CGPoint(
                    x: maskFrame.minX + (CGFloat(x) + 0.5) * maskFrame.width / CGFloat(maskWidth),
                    y: maskFrame.minY + (CGFloat(y) + 0.5) * maskFrame.height / CGFloat(maskHeight)
                )

                guard outerPath.contains(point) else { continue }
                if mouthPath?.contains(point) == true { continue }

                guard let color = LipPixelColor.robustAverage(
                    sampleCluster(point: point, radius: 0.9, previewLayer: previewLayer, sampler: sampler)
                ) else {
                    continue
                }

                let rawScore = lipScore(color, profile: profile)
                let colorProbability = smoothstep((rawScore + threshold * 0.35) / denominator)
                let inCore = corePath.contains(point)
                let geometryFloor: CGFloat = inCore ? 0.36 : 0
                let edgePenalty: CGFloat = inCore ? 1.0 : 0.78
                let pixelAlpha = min(max(max(colorProbability * edgePenalty, geometryFloor), 0), 1)
                alpha[y * maskWidth + x] = UInt8(pixelAlpha * 255)
            }
        }

        alpha = blurred(alpha: alpha, width: maskWidth, height: maskHeight, radius: 1)
        alpha = alpha.map { value in
            let normalized = CGFloat(value) / 255
            let boosted = smoothstep((normalized - 0.12) / 0.72)
            return UInt8(min(max(boosted * 255, 0), 255))
        }

        guard alpha.contains(where: { $0 > 16 }) else { return nil }
        return LipPixelMask(frame: maskFrame, width: maskWidth, height: maskHeight, alpha: alpha)
    }

    private func lipToSkinBoundary(
        around point: CGPoint,
        direction: CGPoint,
        inwardLimit: CGFloat,
        outwardLimit: CGFloat,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler,
        profile: LipColorProfile
    ) -> LipBoundaryResult? {
        var lastLipSample: (point: CGPoint, score: CGFloat)?
        var previousSample: (point: CGPoint, score: CGFloat)?
        var bestDrop: LipBoundaryResult?
        let steps = 16
        let lipThreshold = profile.boundaryThreshold
        let skinThreshold = -profile.boundaryThreshold * 0.55

        for index in 0...steps {
            let progress = CGFloat(index) / CGFloat(steps)
            let offset = -inwardLimit + (inwardLimit + outwardLimit) * progress
            let probePoint = point.offset(by: direction, distance: offset)
            guard let color = LipPixelColor.robustAverage(
                sampleCluster(point: probePoint, radius: 1.4, previewLayer: previewLayer, sampler: sampler)
            ) else {
                continue
            }
            let score = lipScore(color, profile: profile)

            if let previousSample {
                let drop = previousSample.score - score
                if drop > profile.boundaryThreshold,
                   previousSample.score > lipThreshold * 0.25,
                   score < lipThreshold * 0.45 {
                    let confidence = boundaryConfidence(scoreDelta: drop, profile: profile)
                    if confidence > (bestDrop?.confidence ?? 0) {
                        bestDrop = LipBoundaryResult(
                            point: previousSample.point.interpolated(to: probePoint, amount: 0.48),
                            confidence: confidence
                        )
                    }
                }
            }

            if score > lipThreshold {
                lastLipSample = (probePoint, score)
            } else if score < skinThreshold, let lastLipSample {
                let drop = lastLipSample.score - score
                let confidence = boundaryConfidence(scoreDelta: drop, profile: profile)
                guard confidence > 0.18 else { continue }
                return LipBoundaryResult(
                    point: lastLipSample.point.interpolated(to: probePoint, amount: 0.42),
                    confidence: confidence
                )
            }

            previousSample = (probePoint, score)
        }

        guard let bestDrop, bestDrop.confidence > 0.24 else {
            return nil
        }
        return bestDrop
    }

    private func mouthToLipBoundary(
        around point: CGPoint,
        direction: CGPoint,
        mouthSideLimit: CGFloat,
        lipSideLimit: CGFloat,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler,
        profile: LipColorProfile
    ) -> LipBoundaryResult? {
        var previousMouthSample: (point: CGPoint, score: CGFloat)?
        var previousSample: (point: CGPoint, score: CGFloat)?
        var bestRise: LipBoundaryResult?
        let steps = 14
        let lipThreshold = profile.boundaryThreshold
        let mouthThreshold = -profile.boundaryThreshold * 0.40

        for index in 0...steps {
            let progress = CGFloat(index) / CGFloat(steps)
            let offset = -mouthSideLimit + (mouthSideLimit + lipSideLimit) * progress
            let probePoint = point.offset(by: direction, distance: offset)
            guard let color = LipPixelColor.robustAverage(
                sampleCluster(point: probePoint, radius: 1.2, previewLayer: previewLayer, sampler: sampler)
            ) else {
                continue
            }
            let score = lipScore(color, profile: profile)

            if let previousSample {
                let rise = score - previousSample.score
                if rise > profile.boundaryThreshold,
                   previousSample.score < lipThreshold * 0.25,
                   score > lipThreshold * 0.45 {
                    let confidence = boundaryConfidence(scoreDelta: rise, profile: profile)
                    if confidence > (bestRise?.confidence ?? 0) {
                        bestRise = LipBoundaryResult(
                            point: previousSample.point.interpolated(to: probePoint, amount: 0.58),
                            confidence: confidence
                        )
                    }
                }
            }

            if score < mouthThreshold {
                previousMouthSample = (probePoint, score)
            } else if score > lipThreshold, let previousMouthSample {
                let rise = score - previousMouthSample.score
                let confidence = boundaryConfidence(scoreDelta: rise, profile: profile)
                guard confidence > 0.18 else { continue }
                return LipBoundaryResult(
                    point: previousMouthSample.point.interpolated(to: probePoint, amount: 0.58),
                    confidence: confidence
                )
            }

            previousSample = (probePoint, score)
        }

        guard let bestRise, bestRise.confidence > 0.24 else {
            return nil
        }
        return bestRise
    }

    private func normalizedDirection(from start: CGPoint, to end: CGPoint) -> CGPoint? {
        let dx = end.x - start.x
        let dy = end.y - start.y
        let length = hypot(dx, dy)
        guard length > 0.001, length.isFinite else { return nil }
        return CGPoint(x: dx / length, y: dy / length)
    }

    private func averagePoint(_ points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sum = points.reduce(CGPoint.zero) { result, point in
            CGPoint(x: result.x + point.x, y: result.y + point.y)
        }
        let count = CGFloat(points.count)
        return CGPoint(x: sum.x / count, y: sum.y / count)
    }

    private func limitedPoint(
        from point: CGPoint,
        toward target: CGPoint,
        amount: CGFloat,
        maxDistance: CGFloat
    ) -> CGPoint {
        let blendedPoint = point.interpolated(to: target, amount: amount)
        let dx = blendedPoint.x - point.x
        let dy = blendedPoint.y - point.y
        let distance = hypot(dx, dy)
        guard distance > maxDistance, distance > 0.001 else {
            return blendedPoint
        }

        let scale = maxDistance / distance
        return CGPoint(x: point.x + dx * scale, y: point.y + dy * scale)
    }

    private func sample(
        points: [CGPoint],
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler
    ) -> [LipPixelColor] {
        points.compactMap {
            sample(point: $0, previewLayer: previewLayer, sampler: sampler)
        }
    }

    private func sampleCluster(
        points: [CGPoint],
        radius: CGFloat,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler
    ) -> [LipPixelColor] {
        points.flatMap {
            sampleCluster(point: $0, radius: radius, previewLayer: previewLayer, sampler: sampler)
        }
    }

    private func sampleCluster(
        point: CGPoint,
        radius: CGFloat,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler
    ) -> [LipPixelColor] {
        let offsets = [
            CGPoint.zero,
            CGPoint(x: radius, y: 0),
            CGPoint(x: -radius, y: 0),
            CGPoint(x: 0, y: radius),
            CGPoint(x: 0, y: -radius)
        ]

        return offsets.compactMap {
            sample(
                point: CGPoint(x: point.x + $0.x, y: point.y + $0.y),
                previewLayer: previewLayer,
                sampler: sampler
            )
        }
    }

    private func sample(
        point: CGPoint,
        previewLayer: AVCaptureVideoPreviewLayer,
        sampler: LipPixelSampler
    ) -> LipPixelColor? {
        guard previewLayer.bounds.insetBy(dx: -4, dy: -4).contains(point) else {
            return nil
        }

        let capturePoint = previewLayer.captureDevicePointConverted(fromLayerPoint: point)
        return sampler.sample(normalizedX: capturePoint.x, normalizedY: capturePoint.y)
    }

    private func cornerProbePoints(for shape: LipLandmarkShape) -> [CGPoint] {
        guard
            let leftCorner = shape.outerPoints.min(by: { $0.x < $1.x }),
            let rightCorner = shape.outerPoints.max(by: { $0.x < $1.x })
        else {
            return []
        }

        let lipBounds = shape.outerBounds
        let centerY = shape.centerPoint.y
        let step = max(lipBounds.width * 0.055, 4)

        return [
            CGPoint(x: leftCorner.x - step, y: leftCorner.y),
            CGPoint(x: leftCorner.x - step * 1.6, y: leftCorner.y),
            CGPoint(x: leftCorner.x - step, y: centerY),
            CGPoint(x: rightCorner.x + step, y: rightCorner.y),
            CGPoint(x: rightCorner.x + step * 1.6, y: rightCorner.y),
            CGPoint(x: rightCorner.x + step, y: centerY)
        ]
    }

    private func lipLikeRatio(
        samples: [LipPixelColor],
        lipAverage: LipPixelColor,
        skinAverage: LipPixelColor
    ) -> CGFloat {
        guard !samples.isEmpty else { return 0 }

        let lipLikeCount = samples.filter {
            isLipLike($0, lipAverage: lipAverage, skinAverage: skinAverage)
        }.count
        return CGFloat(lipLikeCount) / CGFloat(samples.count)
    }

    private func lipScore(_ color: LipPixelColor, profile: LipColorProfile) -> CGFloat {
        let lipDistance = color.chromaDistance(to: profile.lipAverage)
        let skinDistance = color.chromaDistance(to: profile.skinAverage)
        let redLift = color.normalizedRedness - profile.skinAverage.normalizedRedness
        let saturationLift = color.saturation - profile.skinAverage.saturation
        let luminancePenalty = abs(color.luminance - profile.lipAverage.luminance) * 0.035

        return (skinDistance - lipDistance)
            + redLift * 0.28
            + saturationLift * 0.10
            - luminancePenalty
    }

    private func boundaryConfidence(scoreDelta: CGFloat, profile: LipColorProfile) -> CGFloat {
        let denominator = max(profile.chromaContrast * 1.7, 0.018)
        return min(max(scoreDelta / denominator, 0), 1)
    }

    private func smoothstep(_ value: CGFloat) -> CGFloat {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }

    private func blurred(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        var horizontal = [UInt8](repeating: 0, count: alpha.count)
        var output = [UInt8](repeating: 0, count: alpha.count)

        for y in 0..<height {
            for x in 0..<width {
                var sum = 0
                var count = 0
                for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                    sum += Int(alpha[y * width + sampleX])
                    count += 1
                }
                horizontal[y * width + x] = UInt8(sum / max(count, 1))
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                var sum = 0
                var count = 0
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    sum += Int(horizontal[sampleY * width + x])
                    count += 1
                }
                output[y * width + x] = UInt8(sum / max(count, 1))
            }
        }

        return output
    }

    private func isLipLike(
        _ color: LipPixelColor,
        lipAverage: LipPixelColor,
        skinAverage: LipPixelColor
    ) -> Bool {
        let lipDistance = color.chromaDistance(to: lipAverage)
        let skinDistance = color.chromaDistance(to: skinAverage)
        let redLift = color.normalizedRedness - skinAverage.normalizedRedness
        let saturationLift = color.saturation - skinAverage.saturation
        let score = (skinDistance - lipDistance) + redLift * 0.25 + saturationLift * 0.18
        return score > 0.008
    }
}

private struct LipPixelSampler {
    private let pointer: UnsafeMutablePointer<UInt8>
    private let width: Int
    private let height: Int
    private let bytesPerRow: Int

    init?(pixelBuffer: CVPixelBuffer) {
        guard
            CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
            let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        else {
            return nil
        }

        pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
        width = CVPixelBufferGetWidth(pixelBuffer)
        height = CVPixelBufferGetHeight(pixelBuffer)
        bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    }

    func sample(normalizedX: CGFloat, normalizedY: CGFloat) -> LipPixelColor? {
        guard
            normalizedX.isFinite,
            normalizedY.isFinite,
            normalizedX >= 0,
            normalizedX <= 1,
            normalizedY >= 0,
            normalizedY <= 1
        else {
            return nil
        }

        let x = min(max(Int((normalizedX * CGFloat(width - 1)).rounded()), 0), width - 1)
        let y = min(max(Int((normalizedY * CGFloat(height - 1)).rounded()), 0), height - 1)
        let offset = y * bytesPerRow + x * 4

        return LipPixelColor(
            red: CGFloat(pointer[offset + 2]) / 255,
            green: CGFloat(pointer[offset + 1]) / 255,
            blue: CGFloat(pointer[offset]) / 255
        )
    }
}

private struct LipPixelColor {
    let red: CGFloat
    let green: CGFloat
    let blue: CGFloat

    var saturation: CGFloat {
        let maxValue = max(red, green, blue)
        let minValue = min(red, green, blue)
        return maxValue == 0 ? 0 : (maxValue - minValue) / maxValue
    }

    var luminance: CGFloat {
        red * 0.299 + green * 0.587 + blue * 0.114
    }

    var redness: CGFloat {
        red - ((green + blue) * 0.5)
    }

    var normalizedRedness: CGFloat {
        chromaRed - ((chromaGreen + chromaBlue) * 0.5)
    }

    private var chromaRed: CGFloat {
        red / max(red + green + blue, 0.001)
    }

    private var chromaGreen: CGFloat {
        green / max(red + green + blue, 0.001)
    }

    private var chromaBlue: CGFloat {
        blue / max(red + green + blue, 0.001)
    }

    func distance(to color: LipPixelColor) -> CGFloat {
        let redDelta = red - color.red
        let greenDelta = green - color.green
        let blueDelta = blue - color.blue
        return sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta)
    }

    func chromaDistance(to color: LipPixelColor) -> CGFloat {
        let redDelta = chromaRed - color.chromaRed
        let greenDelta = chromaGreen - color.chromaGreen
        let blueDelta = chromaBlue - color.chromaBlue
        return sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta)
    }

    static func average(_ colors: [LipPixelColor]) -> LipPixelColor? {
        guard !colors.isEmpty else { return nil }

        let sum = colors.reduce((red: CGFloat(0), green: CGFloat(0), blue: CGFloat(0))) { result, color in
            (
                red: result.red + color.red,
                green: result.green + color.green,
                blue: result.blue + color.blue
            )
        }
        let count = CGFloat(colors.count)

        return LipPixelColor(
            red: sum.red / count,
            green: sum.green / count,
            blue: sum.blue / count
        )
    }

    static func robustAverage(_ colors: [LipPixelColor]) -> LipPixelColor? {
        guard !colors.isEmpty else { return nil }
        guard colors.count > 6 else { return average(colors) }

        let sortedByLuminance = colors.sorted { $0.luminance < $1.luminance }
        let trimCount = max(colors.count / 6, 1)
        let trimmed = Array(sortedByLuminance.dropFirst(trimCount).dropLast(trimCount))
        return average(trimmed.isEmpty ? colors : trimmed)
    }
}

private enum LipstickCameraError: LocalizedError {
    case cameraUnavailable(Lip2DCameraPosition)
    case cannotAddCameraInput(Lip2DCameraPosition)
    case cannotAddVideoOutput

    var errorDescription: String? {
        switch self {
        case .cameraUnavailable(let position):
            "找不到\(position.displayName)摄像头"
        case .cannotAddCameraInput(let position):
            "无法接入\(position.displayName)摄像头"
        case .cannotAddVideoOutput:
            "无法读取相机画面"
        }
    }
}

final class LipMakeupOverlayView: UIView {
    private let maskFillLayer = CALayer()
    private let coverageLayer = CAShapeLayer()
    private let fillLayer = CAShapeLayer()
    private let creaseLayer = CAShapeLayer()
    private let edgeLayer = CAShapeLayer()
    private let debugOuterLayer = CAShapeLayer()
    private let debugInnerLayer = CAShapeLayer()
    private let debugPointLayer = CAShapeLayer()
    private var lipColor = UIColor.clear
    private var showsDebugLandmarks = false
    private var latestShape: LipLandmarkShape?
    private var renderedShape: LipLandmarkShape?
    private var latestColorRefinement: LipColorRefinement?
    private var stableSegmentationOffset: LipSegmentationOffset?
    private var lastObservedShape: LipLandmarkShape?
    private var lastObservedTimestamp: CFTimeInterval?
    private var observedMotionOffset = CGPoint.zero
    private var lastRenderTimestamp: CFTimeInterval?
    private var missedFrameCount = 0
    private var segmentationMissCount = 0
    private static let usesColorShapeCorrection = false
    private static let usesMaskTextureOverlay = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = false

        maskFillLayer.opacity = 0.22
        maskFillLayer.isHidden = true
        maskFillLayer.contentsGravity = .resize
        maskFillLayer.compositingFilter = "multiplyBlendMode"
        maskFillLayer.minificationFilter = .linear
        maskFillLayer.magnificationFilter = .linear
        maskFillLayer.actions = Self.disabledActions
        layer.addSublayer(maskFillLayer)

        coverageLayer.fillRule = .evenOdd
        coverageLayer.opacity = 1
        coverageLayer.allowsEdgeAntialiasing = true
        coverageLayer.actions = Self.disabledActions
        layer.addSublayer(coverageLayer)

        fillLayer.fillRule = .evenOdd
        fillLayer.opacity = 1
        fillLayer.allowsEdgeAntialiasing = true
        fillLayer.compositingFilter = "multiplyBlendMode"
        fillLayer.actions = Self.disabledActions
        layer.addSublayer(fillLayer)

        creaseLayer.fillColor = UIColor.clear.cgColor
        creaseLayer.lineCap = .round
        creaseLayer.lineJoin = .round
        creaseLayer.opacity = 1
        creaseLayer.allowsEdgeAntialiasing = true
        creaseLayer.actions = Self.disabledActions
        layer.addSublayer(creaseLayer)

        edgeLayer.fillColor = UIColor.clear.cgColor
        edgeLayer.lineCap = .round
        edgeLayer.lineJoin = .round
        edgeLayer.opacity = 1
        edgeLayer.shadowOffset = .zero
        edgeLayer.allowsEdgeAntialiasing = true
        edgeLayer.actions = Self.disabledActions
        layer.addSublayer(edgeLayer)

        debugOuterLayer.fillColor = UIColor.clear.cgColor
        debugOuterLayer.strokeColor = UIColor.systemCyan.withAlphaComponent(0.92).cgColor
        debugOuterLayer.lineCap = .round
        debugOuterLayer.lineJoin = .round
        debugOuterLayer.isHidden = true
        debugOuterLayer.actions = Self.disabledActions
        layer.addSublayer(debugOuterLayer)

        debugInnerLayer.fillColor = UIColor.clear.cgColor
        debugInnerLayer.strokeColor = UIColor.systemYellow.withAlphaComponent(0.92).cgColor
        debugInnerLayer.lineCap = .round
        debugInnerLayer.lineJoin = .round
        debugInnerLayer.isHidden = true
        debugInnerLayer.actions = Self.disabledActions
        layer.addSublayer(debugInnerLayer)

        debugPointLayer.fillColor = UIColor.white.withAlphaComponent(0.88).cgColor
        debugPointLayer.strokeColor = UIColor.clear.cgColor
        debugPointLayer.isHidden = true
        debugPointLayer.actions = Self.disabledActions
        layer.addSublayer(debugPointLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        coverageLayer.frame = bounds
        maskFillLayer.contentsScale = window?.screen.scale ?? UIScreen.main.scale
        fillLayer.frame = bounds
        creaseLayer.frame = bounds
        edgeLayer.frame = bounds
        debugOuterLayer.frame = bounds
        debugInnerLayer.frame = bounds
        debugPointLayer.frame = bounds
        updateLayerStyles()
    }

    func updateStyle(color: UIColor, showsDebugLandmarks: Bool) {
        lipColor = color
        self.showsDebugLandmarks = showsDebugLandmarks
        debugOuterLayer.isHidden = !showsDebugLandmarks
        debugInnerLayer.isHidden = !showsDebugLandmarks
        debugPointLayer.isHidden = !showsDebugLandmarks
        updateLayerStyles()
    }

    func update(
        shape: LipLandmarkShape?,
        segmentedDisplayShape: LipLandmarkShape? = nil,
        colorRefinement: LipColorRefinement? = nil,
        segmentationConfidence: CGFloat? = nil,
        segmentationMask: LipPixelMask? = nil
    ) {
        guard let shape else {
            missedFrameCount += 1
            guard missedFrameCount > Self.missedFrameLimit else { return }

            latestShape = nil
            renderedShape = nil
            stableSegmentationOffset = nil
            lastObservedShape = nil
            lastObservedTimestamp = nil
            observedMotionOffset = .zero
            maskFillLayer.contents = nil
            maskFillLayer.isHidden = true
            lastRenderTimestamp = nil
            segmentationMissCount = 0
            coverageLayer.path = nil
            fillLayer.path = nil
            creaseLayer.path = nil
            edgeLayer.path = nil
            debugOuterLayer.path = nil
            debugInnerLayer.path = nil
            debugPointLayer.path = nil
            return
        }

        let baseDisplayShape = shape.horizontallyMirrored(in: bounds)
        let correctedShape = segmentationAdjustedShape(
            baseShape: baseDisplayShape,
            segmentedShape: segmentedDisplayShape,
            confidence: segmentationConfidence
        )
        let timestamp = CACurrentMediaTime()
        let displayShape = motionPredictedShape(from: correctedShape, timestamp: timestamp)
        guard displayShape.isPlausible else {
            missedFrameCount += 1
            guard missedFrameCount > Self.missedFrameLimit else { return }

            latestShape = nil
            renderedShape = nil
            stableSegmentationOffset = nil
            lastObservedShape = nil
            lastObservedTimestamp = nil
            observedMotionOffset = .zero
            maskFillLayer.contents = nil
            maskFillLayer.isHidden = true
            lastRenderTimestamp = nil
            segmentationMissCount = 0
            coverageLayer.path = nil
            fillLayer.path = nil
            creaseLayer.path = nil
            edgeLayer.path = nil
            debugOuterLayer.path = nil
            debugInnerLayer.path = nil
            debugPointLayer.path = nil
            return
        }

        missedFrameCount = 0
        let elapsed = lastRenderTimestamp.map { timestamp - $0 } ?? (1.0 / 30.0)
        lastRenderTimestamp = timestamp
        let smoothingAmount = renderedShape?.motionAnchoredSmoothingAmount(
            toward: displayShape,
            elapsed: elapsed
        ) ?? 1
        let stableShape = renderedShape?.motionAnchoredSmoothed(
            toward: displayShape,
            amount: smoothingAmount
        ) ?? displayShape
        latestShape = stableShape
        renderedShape = stableShape
        let refinement: LipColorRefinement
        if let colorRefinement {
            refinement = latestColorRefinement?.smoothed(toward: colorRefinement, amount: 0.16) ?? colorRefinement
        } else {
            refinement = latestColorRefinement ?? .fallback
        }
        latestColorRefinement = refinement

        let innerScale = innerMouthScale(for: stableShape)
        let coveragePath = lipFillPath(
            for: stableShape,
            outerExpansionX: 1.075,
            outerUpperExpansionY: min(max(refinement.upperCoverageY, 0.990), 1.000),
            outerLowerExpansionY: 1.095,
            cornerXBoost: max(refinement.cornerCoverageBoost * 0.95, 0.042),
            innerShrinkX: innerScale.x * 0.86,
            innerShrinkY: innerScale.y * 0.86
        )
        let fillPath = lipFillPath(
            for: stableShape,
            outerExpansionX: 1.032,
            outerUpperExpansionY: min(max(refinement.upperFillY, 0.985), 0.995),
            outerLowerExpansionY: 1.045,
            cornerXBoost: max(refinement.cornerFillBoost * 0.65, 0.026),
            innerShrinkX: innerScale.x * 0.96,
            innerShrinkY: innerScale.y * 0.96
        )
        let featherPath = UIBezierPath.smoothClosedPath(
            points: makeupOuterPoints(
                for: stableShape.cosmeticExpanded(
                    x: 1.080,
                    upperY: min(max(refinement.upperFeatherY, 0.990), 1.000),
                    lowerY: 1.090,
                    cornerXBoost: max(refinement.cornerFeatherBoost * 0.82, 0.036)
                )
            )
        )
        let creasePath = mouthCreasePath(for: stableShape)
        let refinedSegmentationMask: LipPixelMask?
        if Self.usesMaskTextureOverlay, let segmentationMask {
            refinedSegmentationMask = segmentationMask.refined(with: pixelMaskRenderGuide(for: stableShape))
        } else {
            refinedSegmentationMask = nil
        }
        let maskImage = refinedSegmentationMask?.tintedImage(color: lipColor)
        let maskFrame = refinedSegmentationMask.map { pixelMaskFrame(for: stableShape, fallback: $0.frame) }

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        coverageLayer.path = coveragePath.cgPath
        fillLayer.path = fillPath.cgPath
        edgeLayer.path = featherPath.cgPath
        if let maskImage, let maskFrame {
            maskFillLayer.isHidden = false
            maskFillLayer.frame = maskFrame
            maskFillLayer.contents = maskImage
        } else {
            maskFillLayer.isHidden = true
            maskFillLayer.contents = nil
        }
        creaseLayer.path = creasePath?.cgPath
        debugOuterLayer.path = debugOuterPath(for: stableShape).cgPath
        debugInnerLayer.path = debugInnerPath(for: stableShape).cgPath
        debugPointLayer.path = debugPointPath(for: stableShape).cgPath
        updateLayerStyles()
        CATransaction.commit()
    }

    private func motionPredictedShape(from observedShape: LipLandmarkShape, timestamp: CFTimeInterval) -> LipLandmarkShape {
        defer {
            lastObservedShape = observedShape
            lastObservedTimestamp = timestamp
        }

        guard
            let lastObservedShape,
            let lastObservedTimestamp
        else {
            observedMotionOffset = .zero
            return observedShape
        }

        let elapsed = max(timestamp - lastObservedTimestamp, 1.0 / 120.0)
        let previousBounds = lastObservedShape.outerBounds
        let currentBounds = observedShape.outerBounds
        let scale = max(currentBounds.width, currentBounds.height, 1)
        let rawDelta = CGPoint(
            x: currentBounds.midX - previousBounds.midX,
            y: currentBounds.midY - previousBounds.midY
        ).limitedMagnitude(scale * 0.72)
        let leadAmount = min(max(CGFloat(0.045 / elapsed), 0), 1.35)
        let targetOffset = CGPoint(
            x: rawDelta.x * leadAmount,
            y: rawDelta.y * leadAmount * 0.55
        )

        observedMotionOffset = observedMotionOffset.interpolated(to: targetOffset, amount: 0.58)
        let predictedOffset = observedMotionOffset.limitedMagnitude(scale * 0.34)
        return observedShape.translated(by: predictedOffset)
    }

    private func segmentationAdjustedShape(
        baseShape: LipLandmarkShape,
        segmentedShape: LipLandmarkShape?,
        confidence: CGFloat?
    ) -> LipLandmarkShape {
        guard Self.usesColorShapeCorrection else {
            stableSegmentationOffset = nil
            segmentationMissCount = 0
            return baseShape
        }

        guard
            let segmentedShape,
            segmentedShape.isPlausible,
            let confidence,
            confidence > 0.20
        else {
            return segmentationFallbackShape(for: baseShape)
        }

        let maxCorrection = max(baseShape.outerBounds.height * 0.18, 4.5)
        guard let rawOffset = LipSegmentationOffset(
            from: baseShape,
            to: segmentedShape,
            maxDistance: maxCorrection
        ) else {
            return segmentationFallbackShape(for: baseShape)
        }

        segmentationMissCount = 0
        let targetOffset = rawOffset.scaled(by: Self.segmentationInfluence(for: confidence))
        let smoothingAmount = Self.segmentationOffsetSmoothing(for: confidence)
        if let stableSegmentationOffset {
            self.stableSegmentationOffset = stableSegmentationOffset.smoothed(
                toward: targetOffset,
                amount: smoothingAmount
            )
        } else {
            stableSegmentationOffset = targetOffset.scaled(by: 0.72)
        }

        return stableSegmentationOffset?.applied(to: baseShape) ?? baseShape
    }

    private func segmentationFallbackShape(for baseShape: LipLandmarkShape) -> LipLandmarkShape {
        segmentationMissCount += 1
        guard
            let stableSegmentationOffset,
            segmentationMissCount <= Self.segmentationFadeFrameLimit
        else {
            self.stableSegmentationOffset = nil
            return baseShape
        }

        let fade = max(0, 1 - CGFloat(segmentationMissCount) * 0.10)
        let fadedOffset = stableSegmentationOffset.scaled(by: fade)
        self.stableSegmentationOffset = fadedOffset
        return fadedOffset.applied(to: baseShape) ?? baseShape
    }

    private func updateLayerStyles() {
        let alpha = lipColor.cgColor.alpha
        if alpha <= 0.01 {
            coverageLayer.fillColor = UIColor.clear.cgColor
            fillLayer.fillColor = UIColor.clear.cgColor
            creaseLayer.strokeColor = UIColor.clear.cgColor
            edgeLayer.strokeColor = UIColor.clear.cgColor
            edgeLayer.shadowOpacity = 0
        } else {
            coverageLayer.fillColor = lipColor.withAlphaComponent(alpha * 0.24).cgColor
            fillLayer.fillColor = lipColor.withAlphaComponent(alpha * 0.82).cgColor
            creaseLayer.strokeColor = UIColor.black.withAlphaComponent(max(alpha * 0.34, 0.16)).cgColor
            edgeLayer.strokeColor = lipColor.withAlphaComponent(alpha * 0.12).cgColor
            edgeLayer.shadowColor = lipColor.withAlphaComponent(alpha * 0.10).cgColor
            edgeLayer.shadowOpacity = Float(min(max(alpha * 0.18, 0), 0.22))
            edgeLayer.shadowRadius = max(bounds.width * 0.0025, 1.0)
        }
        creaseLayer.lineWidth = max(bounds.width * 0.0055, 1.7)
        edgeLayer.lineWidth = max(bounds.width * 0.006, 1.8)
        debugOuterLayer.lineWidth = max(bounds.width * 0.003, 1.3)
        debugInnerLayer.lineWidth = max(bounds.width * 0.003, 1.3)
    }

    private func lipFillPath(
        for shape: LipLandmarkShape,
        outerExpansionX: CGFloat,
        outerUpperExpansionY: CGFloat,
        outerLowerExpansionY: CGFloat,
        cornerXBoost: CGFloat,
        innerShrinkX: CGFloat,
        innerShrinkY: CGFloat
    ) -> UIBezierPath {
        let path = UIBezierPath()
        let outerShape = shape.cosmeticExpanded(
            x: outerExpansionX,
            upperY: outerUpperExpansionY,
            lowerY: outerLowerExpansionY,
            cornerXBoost: cornerXBoost
        )
        path.append(UIBezierPath.smoothClosedPath(points: makeupOuterPoints(for: outerShape)))

        if let opening = shape.scaledMouthOpening(x: innerShrinkX, y: innerShrinkY) {
            path.append(UIBezierPath.smoothClosedPath(points: makeupInnerPoints(opening)).reversing())
        }

        return path
    }

    private func makeupOuterPoints(for shape: LipLandmarkShape) -> [CGPoint] {
        shape
            .cornerSoftenedOuterPoints(roundness: 0.30)
            .chaikinClosed(iterations: 2)
            .smoothedClosed(amount: 0.16)
    }

    private func makeupInnerPoints(_ points: [CGPoint]) -> [CGPoint] {
        points
            .chaikinClosed(iterations: 1)
            .smoothedClosed(amount: 0.08)
    }

    private func pixelMaskRenderGuide(for shape: LipLandmarkShape) -> LipPixelMask.RenderGuide {
        let clipShape = shape.cosmeticExpanded(
            x: 1.085,
            upperY: 0.970,
            lowerY: 1.090,
            cornerXBoost: 0.046
        )
        let outerPath = UIBezierPath.smoothClosedPath(points: makeupOuterPoints(for: clipShape))
        let innerMouthPath = pixelMaskInnerMouthPath(for: shape)

        return LipPixelMask.RenderGuide(
            outerPath: outerPath,
            innerMouthPath: innerMouthPath,
            upperBoundary: pixelMaskUpperSafetyBoundary(for: shape),
            lowerBoundary: pixelMaskLowerFeatherBoundary(for: shape),
            lipBounds: shape.outerBounds
        )
    }

    private func pixelMaskUpperSafetyBoundary(for shape: LipLandmarkShape) -> [CGPoint]? {
        let points = shape.outerPoints
        guard points.count >= 11 else { return nil }

        let bounds = shape.outerBounds
        let center = shape.centerPoint
        let halfWidth = max(bounds.width * 0.5, 1)
        let baseInset = max(bounds.height * 0.042, 1.1)
        let upperPoints = Array(points[0...10])

        return upperPoints.map { point in
            let centerWeight = max(0, 1 - abs(point.x - center.x) / halfWidth)
            let verticalInset = baseInset * (0.48 + centerWeight * 0.62)
            return CGPoint(x: point.x, y: point.y + verticalInset)
        }
    }

    private func pixelMaskLowerFeatherBoundary(for shape: LipLandmarkShape) -> [CGPoint]? {
        let points = shape.outerPoints
        guard points.count >= 20 else { return nil }

        let lowerPoints = Array(points[10...19]) + [points[0]]
        return lowerPoints.map { point in
            CGPoint(x: point.x, y: point.y + max(shape.outerBounds.height * 0.012, 0.50))
        }
    }

    private func pixelMaskInnerMouthPath(for shape: LipLandmarkShape) -> UIBezierPath? {
        guard
            shape.mouthOpeningRatio > 0.055,
            let opening = shape.scaledMouthOpening(x: 0.92, y: 0.88)
        else {
            return nil
        }

        return UIBezierPath.smoothClosedPath(points: makeupInnerPoints(opening))
    }

    private func pixelMaskFrame(for shape: LipLandmarkShape, fallback: CGRect) -> CGRect {
        let stableFrame = shape.outerBounds
            .expandedBy(widthRatio: 0.20, heightRatio: 0.36)
            .intersection(bounds)

        if fallback.width > stableFrame.width * 1.8 || fallback.height > stableFrame.height * 1.8 {
            if let currentFaceFrame = LipParsingModel.faceROI(for: shape, in: bounds) {
                return currentFaceFrame
            }

            let faceFrame = fallback.intersection(bounds)
            if faceFrame.width > 12, faceFrame.height > 8 {
                return faceFrame
            }
        }

        guard stableFrame.width > 12, stableFrame.height > 8 else {
            return fallback
        }

        return stableFrame
    }

    private func innerMouthScale(for shape: LipLandmarkShape) -> CGPoint {
        let openingRatio = shape.mouthOpeningRatio

        if openingRatio > 0.42 {
            return CGPoint(x: 1.02, y: 1.04)
        }

        if openingRatio > 0.30 {
            return CGPoint(x: 0.96, y: 0.96)
        }

        if openingRatio > 0.18 {
            return CGPoint(x: 0.86, y: 0.80)
        }

        return CGPoint(x: 0.62, y: 0.44)
    }

    private func mouthCreasePath(for shape: LipLandmarkShape) -> UIBezierPath? {
        guard let innerPoints = shape.innerPoints, innerPoints.count >= 8 else { return nil }

        if shape.mouthOpeningRatio > 0.16 {
            return UIBezierPath.smoothClosedPath(points: shape.scaledMouthOpening(x: 0.96, y: 0.98) ?? innerPoints)
        }

        let halfCount = innerPoints.count / 2
        let creasePoints = (0..<halfCount).map { index in
            let upperPoint = innerPoints[index]
            let lowerPoint = innerPoints[innerPoints.count - 1 - index]
            return upperPoint.interpolated(to: lowerPoint, amount: 0.52)
        }

        guard creasePoints.count >= 3 else { return nil }
        return UIBezierPath.smoothOpenPath(points: creasePoints)
    }

    private func debugOuterPath(for shape: LipLandmarkShape) -> UIBezierPath {
        UIBezierPath.smoothClosedPath(points: shape.outerPoints)
    }

    private func debugInnerPath(for shape: LipLandmarkShape) -> UIBezierPath {
        let path = UIBezierPath()
        if let innerPoints = shape.innerPoints {
            path.append(UIBezierPath.smoothClosedPath(points: innerPoints))
        }
        return path
    }

    private func debugPointPath(for shape: LipLandmarkShape) -> UIBezierPath {
        let path = UIBezierPath()
        for point in shape.debugPoints {
            path.append(
                UIBezierPath(
                    ovalIn: CGRect(
                        x: point.x - 1.7,
                        y: point.y - 1.7,
                        width: 3.4,
                        height: 3.4
                    )
                )
            )
        }

        return path
    }

    private static let disabledActions: [String: CAAction] = [
        "bounds": NSNull(),
        "contents": NSNull(),
        "fillColor": NSNull(),
        "frame": NSNull(),
        "hidden": NSNull(),
        "lineWidth": NSNull(),
        "opacity": NSNull(),
        "path": NSNull(),
        "position": NSNull(),
        "shadowColor": NSNull(),
        "shadowOpacity": NSNull(),
        "shadowRadius": NSNull(),
        "strokeColor": NSNull()
    ]

    private static let missedFrameLimit = 60
    private static let segmentationFadeFrameLimit = 8

    private static func segmentationInfluence(for confidence: CGFloat) -> CGFloat {
        let clampedConfidence = min(max(confidence, 0), 1)
        return 0.28 + clampedConfidence * 0.48
    }

    private static func segmentationOffsetSmoothing(for confidence: CGFloat) -> CGFloat {
        let clampedConfidence = min(max(confidence, 0), 1)
        return 0.10 + clampedConfidence * 0.16
    }
}

struct LipLandmarkShape {
    let outerPoints: [CGPoint]
    let innerPoints: [CGPoint]?
    let debugPoints: [CGPoint]
    let mediaPipeProjection: MediaPipeProjection?

    var isPlausible: Bool {
        let bounds = Self.bounds(for: outerPoints)
        guard bounds.width > 20, bounds.height > 6 else { return false }
        return bounds.width > bounds.height * 0.72
    }

    var outerBounds: CGRect {
        Self.bounds(for: outerPoints)
    }

    var centerPoint: CGPoint {
        Self.center(of: outerPoints)
    }

    var reducedMouthOpening: [CGPoint] {
        guard let innerPoints, innerPoints.count >= 6 else { return [] }
        return Self.scaled(points: innerPoints, x: 0.64, y: 0.52)
    }

    var mouthOpeningRatio: CGFloat {
        guard let innerPoints, innerPoints.count >= 6 else { return 0 }
        let outerBounds = Self.bounds(for: outerPoints)
        let innerBounds = Self.bounds(for: innerPoints)
        return innerBounds.height / max(outerBounds.height, 1)
    }

    init?(visionObservation observation: VNFaceObservation, previewLayer: AVCaptureVideoPreviewLayer) {
        guard let outerLips = observation.landmarks?.outerLips else { return nil }

        let outerPoints = Self.visionPoints(
            from: outerLips,
            observation: observation,
            previewLayer: previewLayer
        )

        guard outerPoints.count >= 6 else { return nil }

        self.outerPoints = outerPoints
        if let innerLips = observation.landmarks?.innerLips {
            let innerPoints = Self.visionPoints(
                from: innerLips,
                observation: observation,
                previewLayer: previewLayer
            )
            self.innerPoints = innerPoints.count >= 6 ? innerPoints : nil
        } else {
            self.innerPoints = nil
        }
        debugPoints = outerPoints + (innerPoints ?? [])
        mediaPipeProjection = nil
    }

    #if canImport(MediaPipeTasksVision)
    init?(
        mediaPipeLandmarks landmarks: [NormalizedLandmark],
        previewLayer: AVCaptureVideoPreviewLayer,
        anchorShape: LipLandmarkShape?,
        preferredProjection: MediaPipeProjection?
    ) {
        let viewBounds = previewLayer.bounds
        let outerLandmarks = Self.normalizedMediaPipePoints(
            from: Self.outerLipIndices,
            landmarks: landmarks
        )
        let innerLandmarks = Self.normalizedMediaPipePoints(
            from: Self.innerLipIndices,
            landmarks: landmarks
        )

        guard
            outerLandmarks.count == Self.outerLipIndices.count,
            innerLandmarks.count == Self.innerLipIndices.count
        else {
            return nil
        }

        let candidates = MediaPipeProjection.allCases.map { projection in
            let outerPoints = outerLandmarks.map {
                Self.viewPoint(from: projection.map($0), previewLayer: previewLayer)
            }
            let innerPoints = innerLandmarks.map {
                Self.viewPoint(from: projection.map($0), previewLayer: previewLayer)
            }
            let geometryScore = Self.projectionScore(points: outerPoints, viewBounds: viewBounds)
            return (
                projection: projection,
                outerPoints: outerPoints,
                innerPoints: innerPoints,
                geometryScore: geometryScore,
                anchorScore: anchorShape.map {
                    Self.anchorProjectionScore(points: outerPoints, anchorShape: $0)
                } ?? -1000
            )
        }

        let preferredCandidate = preferredProjection.flatMap { projection in
            candidates.first { $0.projection == projection && $0.geometryScore > 0 }
        }
        let anchoredCandidate = candidates.max(by: { $0.anchorScore < $1.anchorScore })
        let geometryCandidate = candidates.max(by: { $0.geometryScore < $1.geometryScore })
        let best: (projection: MediaPipeProjection, outerPoints: [CGPoint], innerPoints: [CGPoint], geometryScore: CGFloat, anchorScore: CGFloat)?

        if anchorShape != nil, let anchoredCandidate, anchoredCandidate.anchorScore > 0 {
            if
                let preferredCandidate,
                preferredCandidate.anchorScore >= anchoredCandidate.anchorScore - Self.projectionSwitchTolerance
            {
                best = preferredCandidate
            } else {
                best = anchoredCandidate
            }
        } else if let preferredCandidate {
            best = preferredCandidate
        } else if let geometryCandidate, geometryCandidate.geometryScore > 0 {
            best = geometryCandidate
        } else {
            best = nil
        }

        guard let best else {
            return nil
        }

        self.outerPoints = best.outerPoints
        self.innerPoints = best.innerPoints
        debugPoints = best.outerPoints + best.innerPoints
        mediaPipeProjection = best.projection
    }
    #endif

    private static let outerLipIndices = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146
    ]

    private static let innerLipIndices = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]

    private static let projectionSwitchTolerance: CGFloat = 70

    private static func visionPoints(
        from region: VNFaceLandmarkRegion2D,
        observation: VNFaceObservation,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> [CGPoint] {
        region.normalizedPoints.map { point in
            let normalized = CGPoint(
                x: observation.boundingBox.minX + point.x * observation.boundingBox.width,
                y: observation.boundingBox.minY + point.y * observation.boundingBox.height
            )
            return previewLayer.layerPointConverted(
                fromCaptureDevicePoint: CGPoint(x: normalized.x, y: 1 - normalized.y)
            )
        }
    }

    #if canImport(MediaPipeTasksVision)
    private static func normalizedMediaPipePoints(
        from indices: [Int],
        landmarks: [NormalizedLandmark]
    ) -> [CGPoint] {
        indices.compactMap { index in
            guard landmarks.indices.contains(index) else { return nil }
            let landmark = landmarks[index]
            return CGPoint(x: CGFloat(landmark.x), y: CGFloat(landmark.y))
        }
    }

    private static func viewPoint(
        from normalizedPoint: CGPoint,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> CGPoint {
        previewLayer.layerPointConverted(
            fromCaptureDevicePoint: CGPoint(
                x: min(max(normalizedPoint.x, 0), 1),
                y: min(max(normalizedPoint.y, 0), 1)
            )
        )
    }

    private static func projectionScore(points: [CGPoint], viewBounds: CGRect) -> CGFloat {
        let bounds = bounds(for: points)
        let width = bounds.width
        let height = bounds.height
        let aspect = width / max(height, 1)
        let centerX = bounds.midX / max(viewBounds.width, 1)
        let centerY = bounds.midY / max(viewBounds.height, 1)

        guard width > viewBounds.width * 0.035, height > viewBounds.height * 0.006 else {
            return -1000
        }

        guard width < viewBounds.width * 0.92, height < viewBounds.height * 0.36 else {
            return -1000
        }

        guard aspect > 0.70 else {
            return -1000
        }

        var score: CGFloat = 120
        score -= abs(aspect - 2.20) * 10
        score -= abs(centerX - 0.5) * 36

        if centerY < 0.28 {
            score -= (0.28 - centerY) * 120
        }

        if centerY > 0.88 {
            score -= (centerY - 0.88) * 120
        }

        return score
    }

    private static func anchorProjectionScore(points: [CGPoint], anchorShape: LipLandmarkShape) -> CGFloat {
        let pointBounds = bounds(for: points)
        let anchorBounds = bounds(for: anchorShape.outerPoints)

        guard pointBounds.width > 1, pointBounds.height > 1, anchorBounds.width > 1, anchorBounds.height > 1 else {
            return -1000
        }

        let overlap = intersectionOverUnion(pointBounds, anchorBounds)
        guard overlap > 0.06 else { return -1000 }

        let centerDistance = hypot(pointBounds.midX - anchorBounds.midX, pointBounds.midY - anchorBounds.midY)
        let anchorDiagonal = hypot(anchorBounds.width, anchorBounds.height)
        let normalizedCenterDistance = centerDistance / max(anchorDiagonal, 1)
        let widthRatio = pointBounds.width / max(anchorBounds.width, 1)
        let heightRatio = pointBounds.height / max(anchorBounds.height, 1)
        let sizePenalty = abs(log(widthRatio)) + abs(log(heightRatio))

        var score: CGFloat = 500
        score += overlap * 420
        score -= normalizedCenterDistance * 260
        score -= sizePenalty * 58
        return score
    }

    private static func intersectionOverUnion(_ rectA: CGRect, _ rectB: CGRect) -> CGFloat {
        let intersection = rectA.intersection(rectB)
        guard !intersection.isNull else { return 0 }

        let intersectionArea = intersection.width * intersection.height
        let unionArea = rectA.width * rectA.height + rectB.width * rectB.height - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
    #endif

    func smoothed(toward shape: LipLandmarkShape, amount: CGFloat) -> LipLandmarkShape {
        guard outerPoints.count == shape.outerPoints.count else { return shape }

        let innerPoints: [CGPoint]?
        if
            let currentInner = self.innerPoints,
            let nextInner = shape.innerPoints,
            currentInner.count == nextInner.count
        {
            innerPoints = currentInner.indices.map {
                currentInner[$0].interpolated(to: nextInner[$0], amount: amount)
            }
        } else {
            innerPoints = shape.innerPoints
        }

        let outerPoints = outerPoints.indices.map {
            self.outerPoints[$0].interpolated(to: shape.outerPoints[$0], amount: amount)
        }

        return LipLandmarkShape(
            outerPoints: outerPoints,
            innerPoints: innerPoints,
            mediaPipeProjection: mediaPipeProjection
        )
    }

    func motionAnchoredSmoothed(toward shape: LipLandmarkShape, amount: CGFloat) -> LipLandmarkShape {
        guard outerPoints.count == shape.outerPoints.count else { return shape }

        let currentBounds = Self.bounds(for: outerPoints)
        let targetBounds = Self.bounds(for: shape.outerPoints)
        let anchoredOuterPoints = Self.motionAnchored(
            points: outerPoints,
            from: currentBounds,
            to: targetBounds
        )
        let maxContourStep = max(targetBounds.height * (amount > 0.42 ? 0.32 : 0.22), 4.5)
        let deadzone = max(targetBounds.height * 0.018, 0.75)
        let outerPoints = anchoredOuterPoints.indices.map { index in
            let limitedTarget = anchoredOuterPoints[index].limitedToward(
                shape.outerPoints[index],
                maxDistance: maxContourStep,
                deadzone: deadzone
            )
            return anchoredOuterPoints[index].interpolated(to: limitedTarget, amount: amount)
        }

        let innerPoints: [CGPoint]?
        if
            let currentInner = self.innerPoints,
            let nextInner = shape.innerPoints,
            currentInner.count == nextInner.count
        {
            let currentInnerBounds = Self.bounds(for: currentInner)
            let nextInnerBounds = Self.bounds(for: nextInner)
            let anchoredInnerPoints = Self.motionAnchored(
                points: currentInner,
                from: currentInnerBounds,
                to: nextInnerBounds
            )
            let maxInnerStep = max(targetBounds.height * (amount > 0.42 ? 0.28 : 0.18), 3.5)
            let innerDeadzone = max(targetBounds.height * 0.015, 0.65)
            innerPoints = anchoredInnerPoints.indices.map { index in
                let limitedTarget = anchoredInnerPoints[index].limitedToward(
                    nextInner[index],
                    maxDistance: maxInnerStep,
                    deadzone: innerDeadzone
                )
                return anchoredInnerPoints[index].interpolated(to: limitedTarget, amount: amount)
            }
        } else {
            innerPoints = shape.innerPoints
        }

        return LipLandmarkShape(
            outerPoints: outerPoints,
            innerPoints: innerPoints,
            mediaPipeProjection: shape.mediaPipeProjection ?? mediaPipeProjection
        )
    }

    func replacing(outerPoints newOuterPoints: [CGPoint]? = nil, innerPoints newInnerPoints: [CGPoint]? = nil) -> LipLandmarkShape {
        LipLandmarkShape(
            outerPoints: newOuterPoints ?? outerPoints,
            innerPoints: newInnerPoints ?? innerPoints,
            mediaPipeProjection: mediaPipeProjection
        )
    }

    func translated(by offset: CGPoint) -> LipLandmarkShape {
        LipLandmarkShape(
            outerPoints: outerPoints.map { $0.applying(offset: offset) },
            innerPoints: innerPoints?.map { $0.applying(offset: offset) },
            mediaPipeProjection: mediaPipeProjection
        )
    }

    func trackingSmoothingAmount(toward shape: LipLandmarkShape, elapsed: TimeInterval) -> CGFloat {
        let currentBounds = Self.bounds(for: outerPoints)
        let nextBounds = Self.bounds(for: shape.outerPoints)
        let movement = hypot(currentBounds.midX - nextBounds.midX, currentBounds.midY - nextBounds.midY)
        let scale = max(currentBounds.width, currentBounds.height, 1)
        let normalizedMovement = movement / scale
        let speed = normalizedMovement / CGFloat(max(elapsed, 1.0 / 120.0))

        if speed > 9 {
            return 1
        }

        if speed > 5 {
            return 0.97
        }

        if movement > scale * 0.40 {
            return 0.98
        }

        if movement > scale * 0.18 {
            return 0.92
        }

        if movement > scale * 0.07 {
            return 0.84
        }

        return 0.72
    }

    func motionAnchoredSmoothingAmount(toward shape: LipLandmarkShape, elapsed: TimeInterval) -> CGFloat {
        let currentBounds = Self.bounds(for: outerPoints)
        let nextBounds = Self.bounds(for: shape.outerPoints)
        let movement = hypot(currentBounds.midX - nextBounds.midX, currentBounds.midY - nextBounds.midY)
        let scale = max(nextBounds.width, nextBounds.height, 1)
        let normalizedMovement = movement / scale
        let speed = normalizedMovement / CGFloat(max(elapsed, 1.0 / 120.0))
        let contourDelta = motionAnchoredContourDelta(toward: shape)
        let openingDelta = abs(mouthOpeningRatio - shape.mouthOpeningRatio)
        let sizeDelta = motionAnchoredSizeDelta(toward: shape)

        if contourDelta < 0.018, openingDelta < 0.025, sizeDelta < 0.030 {
            return 0.12
        }

        if openingDelta > 0.20 {
            return 0.92
        }

        if openingDelta > 0.10 {
            return 0.82
        }

        if sizeDelta > 0.20 {
            return 0.86
        }

        if sizeDelta > 0.10 {
            return 0.72
        }

        if contourDelta > 0.34 {
            return 0.82
        }

        if contourDelta > 0.22 {
            return 0.68
        }

        if speed > 7 {
            return 0.84
        }

        if speed > 3 {
            return 0.64
        }

        return 0.28
    }

    func allowsTransition(to shape: LipLandmarkShape) -> Bool {
        let currentBounds = Self.bounds(for: outerPoints)
        let nextBounds = Self.bounds(for: shape.outerPoints)
        let movement = hypot(currentBounds.midX - nextBounds.midX, currentBounds.midY - nextBounds.midY)
        let scale = max(currentBounds.width, currentBounds.height, 1)
        let widthRatio = nextBounds.width / max(currentBounds.width, 1)
        let heightRatio = nextBounds.height / max(currentBounds.height, 1)

        guard movement < scale * 1.35 else { return false }
        guard widthRatio > 0.32, widthRatio < 2.50 else { return false }
        guard heightRatio > 0.30, heightRatio < 3.50 else { return false }
        return true
    }

    private func motionAnchoredContourDelta(toward shape: LipLandmarkShape) -> CGFloat {
        guard outerPoints.count == shape.outerPoints.count else { return 1 }

        let currentBounds = Self.bounds(for: outerPoints)
        let nextBounds = Self.bounds(for: shape.outerPoints)
        let anchoredOuterPoints = Self.motionAnchored(
            points: outerPoints,
            from: currentBounds,
            to: nextBounds
        )
        let totalDelta = anchoredOuterPoints.indices.reduce(CGFloat(0)) { result, index in
            result + anchoredOuterPoints[index].distance(to: shape.outerPoints[index])
        }
        let scale = max(nextBounds.width, nextBounds.height, 1)
        return (totalDelta / CGFloat(max(anchoredOuterPoints.count, 1))) / scale
    }

    private func motionAnchoredSizeDelta(toward shape: LipLandmarkShape) -> CGFloat {
        let currentBounds = Self.bounds(for: outerPoints)
        let nextBounds = Self.bounds(for: shape.outerPoints)
        let widthRatio = nextBounds.width / max(currentBounds.width, 1)
        let heightRatio = nextBounds.height / max(currentBounds.height, 1)
        return abs(log(widthRatio)) + abs(log(heightRatio))
    }

    func cosmeticExpanded(x: CGFloat, y: CGFloat) -> LipLandmarkShape {
        LipLandmarkShape(
            outerPoints: Self.scaled(points: outerPoints, x: x, y: y),
            innerPoints: innerPoints,
            mediaPipeProjection: mediaPipeProjection
        )
    }

    func cosmeticExpanded(
        x: CGFloat,
        upperY: CGFloat,
        lowerY: CGFloat,
        cornerXBoost: CGFloat
    ) -> LipLandmarkShape {
        let center = Self.center(of: outerPoints)
        let lipBounds = Self.bounds(for: outerPoints)
        let cornerStart = lipBounds.width * 0.34
        let cornerRange = max(lipBounds.width * 0.16, 1)

        let expandedOuterPoints = outerPoints.map { point in
            let dx = point.x - center.x
            let dy = point.y - center.y
            let rawCornerWeight = min(max((abs(dx) - cornerStart) / cornerRange, 0), 1)
            let cornerWeight = rawCornerWeight * rawCornerWeight * (3 - 2 * rawCornerWeight) * 0.70
            let xScale = x + cornerXBoost * cornerWeight
            let yScale = dy < 0 ? upperY : lowerY

            return CGPoint(
                x: center.x + dx * xScale,
                y: center.y + dy * yScale
            )
        }

        return LipLandmarkShape(
            outerPoints: expandedOuterPoints,
            innerPoints: innerPoints,
            mediaPipeProjection: mediaPipeProjection
        )
    }

    func cornerSoftenedOuterPoints(roundness: CGFloat = 0.34) -> [CGPoint] {
        guard outerPoints.count >= 8 else { return outerPoints }

        let leftCornerIndex = outerPoints.indices.min { outerPoints[$0].x < outerPoints[$1].x }
        let rightCornerIndex = outerPoints.indices.max { outerPoints[$0].x < outerPoints[$1].x }
        let cornerIndices = Set([leftCornerIndex, rightCornerIndex].compactMap { $0 })

        return outerPoints.indices.flatMap { index -> [CGPoint] in
            guard cornerIndices.contains(index) else {
                return [outerPoints[index]]
            }

            let previous = outerPoints[(index - 1 + outerPoints.count) % outerPoints.count]
            let current = outerPoints[index]
            let next = outerPoints[(index + 1) % outerPoints.count]

            return [
                current.interpolated(to: previous, amount: roundness),
                current.interpolated(to: next, amount: roundness)
            ]
        }
    }

    func scaledMouthOpening(x: CGFloat, y: CGFloat) -> [CGPoint]? {
        guard let innerPoints, innerPoints.count >= 6 else { return nil }
        return Self.scaled(points: innerPoints, x: x, y: y)
    }

    func horizontallyMirrored(in rect: CGRect) -> LipLandmarkShape {
        guard rect.width > 1 else { return self }

        func mirror(_ point: CGPoint) -> CGPoint {
            CGPoint(
                x: rect.minX + rect.maxX - point.x,
                y: point.y
            )
        }

        return LipLandmarkShape(
            outerPoints: outerPoints.map(mirror),
            innerPoints: innerPoints?.map(mirror),
            mediaPipeProjection: mediaPipeProjection
        )
    }

    private init(
        outerPoints: [CGPoint],
        innerPoints: [CGPoint]?,
        mediaPipeProjection: MediaPipeProjection? = nil
    ) {
        self.outerPoints = outerPoints
        self.innerPoints = innerPoints
        debugPoints = outerPoints + (innerPoints ?? [])
        self.mediaPipeProjection = mediaPipeProjection
    }

    private static func scaled(points: [CGPoint], x: CGFloat, y: CGFloat) -> [CGPoint] {
        let center = center(of: points)
        return points.map { point in
            CGPoint(
                x: center.x + (point.x - center.x) * x,
                y: center.y + (point.y - center.y) * y
            )
        }
    }

    private static func motionAnchored(points: [CGPoint], from currentBounds: CGRect, to targetBounds: CGRect) -> [CGPoint] {
        guard
            currentBounds.width > 0.001,
            currentBounds.height > 0.001,
            targetBounds.width > 0.001,
            targetBounds.height > 0.001
        else {
            return points
        }

        let scaleX = targetBounds.width / currentBounds.width
        let scaleY = targetBounds.height / currentBounds.height

        return points.map { point in
            CGPoint(
                x: targetBounds.midX + (point.x - currentBounds.midX) * scaleX,
                y: targetBounds.midY + (point.y - currentBounds.midY) * scaleY
            )
        }
    }

    private static func center(of points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sum = points.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }
        let count = CGFloat(points.count)
        return CGPoint(x: sum.x / count, y: sum.y / count)
    }

    private static func bounds(for points: [CGPoint]) -> CGRect {
        guard let first = points.first else { return .zero }
        return points.dropFirst().reduce(CGRect(origin: first, size: .zero)) { rect, point in
            rect.union(CGRect(origin: point, size: .zero))
        }
    }

    enum MediaPipeProjection: CaseIterable {
        case identity
        case mirrorX
        case mirrorY
        case rotate180
        case transpose
        case transposeMirrorX
        case transposeMirrorY
        case antiTranspose

        func map(_ point: CGPoint) -> CGPoint {
            switch self {
            case .identity:
                point
            case .mirrorX:
                CGPoint(x: 1 - point.x, y: point.y)
            case .mirrorY:
                CGPoint(x: point.x, y: 1 - point.y)
            case .rotate180:
                CGPoint(x: 1 - point.x, y: 1 - point.y)
            case .transpose:
                CGPoint(x: point.y, y: point.x)
            case .transposeMirrorX:
                CGPoint(x: 1 - point.y, y: point.x)
            case .transposeMirrorY:
                CGPoint(x: point.y, y: 1 - point.x)
            case .antiTranspose:
                CGPoint(x: 1 - point.y, y: 1 - point.x)
            }
        }
    }
}

private extension UIBezierPath {
    static func smoothOpenPath(points: [CGPoint]) -> UIBezierPath {
        let path = UIBezierPath()
        guard let first = points.first else { return path }

        path.move(to: first)

        guard points.count > 2 else {
            for point in points.dropFirst() {
                path.addLine(to: point)
            }
            return path
        }

        for index in points.indices.dropFirst() {
            let previous = points[index - 1]
            let current = points[index]
            let next = index + 1 < points.count ? points[index + 1] : current
            let controlA = previous.interpolated(to: current, amount: 0.58)
            let controlB = next.interpolated(to: current, amount: 0.58)
            path.addCurve(to: current, controlPoint1: controlA, controlPoint2: controlB)
        }

        return path
    }

    static func smoothClosedPath(points: [CGPoint]) -> UIBezierPath {
        let path = UIBezierPath()
        guard let first = points.first else { return path }

        path.move(to: first)

        for index in points.indices.dropFirst() {
            let previous = points[(index - 1 + points.count) % points.count]
            let current = points[index]
            let next = points[(index + 1) % points.count]

            let controlA = previous.interpolated(to: current, amount: 0.58)
            let controlB = next.interpolated(to: current, amount: 0.58)
            path.addCurve(to: current, controlPoint1: controlA, controlPoint2: controlB)
        }

        let last = points[points.count - 1]
        let second = points.count > 1 ? points[1] : first
        path.addCurve(
            to: first,
            controlPoint1: last.interpolated(to: first, amount: 0.58),
            controlPoint2: second.interpolated(to: first, amount: 0.58)
        )
        path.close()
        return path
    }
}

private extension Array where Element == CGPoint {
    func chaikinClosed(iterations: Int) -> [CGPoint] {
        guard count >= 3, iterations > 0 else { return self }

        var points = self
        for _ in 0..<iterations {
            var refined: [CGPoint] = []
            refined.reserveCapacity(points.count * 2)

            for index in points.indices {
                let current = points[index]
                let next = points[(index + 1) % points.count]
                refined.append(current.interpolated(to: next, amount: 0.25))
                refined.append(current.interpolated(to: next, amount: 0.75))
            }

            points = refined
        }

        return points
    }

    func smoothedClosed(amount: CGFloat) -> [CGPoint] {
        guard count >= 3 else { return self }

        return indices.map { index in
            let previous = self[(index - 1 + count) % count]
            let current = self[index]
            let next = self[(index + 1) % count]
            let neighborMidpoint = CGPoint(
                x: (previous.x + next.x) * 0.5,
                y: (previous.y + next.y) * 0.5
            )
            return current.interpolated(to: neighborMidpoint, amount: amount)
        }
    }
}

private extension CGRect {
    func expandedBy(widthRatio: CGFloat, heightRatio: CGFloat) -> CGRect {
        insetBy(dx: -width * widthRatio, dy: -height * heightRatio)
    }
}

private final class Lip2DTrainingSampleRecorder {
    var onStatusChange: ((String) -> Void)?
    var onRecordingActiveChange: ((Bool) -> Void)?
    var cameraPosition: Lip2DCameraPosition = .back

    private let queue = DispatchQueue(label: "com.perfectmakeup.lipstick.2d-recorder", qos: .utility)
    private let lock = NSLock()
    private let ciContext = CIContext()
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private let minimumFrameInterval: CFTimeInterval = 1.0 / 5.0
    private let guide = ARKitUVRecordingGuide.default

    private var isRecording = false
    private var sessionURL: URL?
    private var sessionName = ""
    private var nextSampleIndex = 0
    private var completedSampleCount = 0
    private var lastSampleTime: CFTimeInterval = 0
    private var recordingStartedAt: CFTimeInterval = 0
    private var guideTimer: DispatchSourceTimer?

    func start() {
        lock.lock()
        if isRecording {
            lock.unlock()
            return
        }
        lock.unlock()

        do {
            let rootURL = try Self.recordingsRootURL()
            try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)

            let sessionName = Self.sessionName()
            let sessionURL = rootURL.appendingPathComponent(sessionName, isDirectory: true)
            try FileManager.default.createDirectory(at: sessionURL.appendingPathComponent("images", isDirectory: true), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: sessionURL.appendingPathComponent("frames", isDirectory: true), withIntermediateDirectories: true)

            let session = Lip2DSessionMetadata(
                schemaVersion: 1,
                createdAt: ISO8601DateFormatter().string(from: Date()),
                coordinateSystem: "AVCaptureVideoDataOutput \(cameraPosition.rawValue)-camera pixel buffer",
                notes: [
                    "This recording is independent of ARKit face tracking.",
                    "Use these frames for SAM2/CVAT/manual lip boundary labeling.",
                    cameraPosition.previewMirrored
                        ? "Preview is mirrored for the user, saved frames use the capture buffer orientation."
                        : "Preview and saved frames use the back-camera capture orientation."
                ]
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(session).write(to: sessionURL.appendingPathComponent("session.json"), options: .atomic)

            lock.lock()
            self.isRecording = true
            self.sessionURL = sessionURL
            self.sessionName = sessionName
            self.nextSampleIndex = 0
            self.completedSampleCount = 0
            self.lastSampleTime = 0
            self.recordingStartedAt = CACurrentMediaTime()
            lock.unlock()

            publishRecordingActive(true)
            startGuideTimer()
            publishGuideStatus()
        } catch {
            publishStatus("2D REC failed: \(error.localizedDescription)")
            publishRecordingActive(false)
        }
    }

    func stop() {
        stop(completedAutomatically: false)
    }

    private func stop(completedAutomatically: Bool) {
        cancelGuideTimer()

        lock.lock()
        guard isRecording else {
            lock.unlock()
            publishStatus("")
            publishRecordingActive(false)
            return
        }
        let count = completedSampleCount
        isRecording = false
        sessionURL = nil
        sessionName = ""
        lock.unlock()

        publishStatus(completedAutomatically ? "2D REC 完成 saved \(count)" : "2D REC saved \(count)")
        publishRecordingActive(false)
    }

    func record(sampleBuffer: CMSampleBuffer) {
        let now = CACurrentMediaTime()

        lock.lock()
        guard isRecording, now - lastSampleTime >= minimumFrameInterval, let sessionURL else {
            lock.unlock()
            return
        }
        let sampleIndex = nextSampleIndex
        let elapsedSeconds = self.elapsedGuideSeconds(now: now)
        let guideState = guide.state(atElapsedSeconds: elapsedSeconds)
        nextSampleIndex += 1
        lastSampleTime = now
        lock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let sampleID = String(format: "frame_%06d", sampleIndex)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let payload = Lip2DFrameMetadata(
            schemaVersion: 1,
            sampleID: sampleID,
            timestamp: timestamp.isFinite ? timestamp : 0,
            image: Lip2DImageMetadata(
                file: "images/\(sampleID).jpg",
                width: width,
                height: height,
                pixelFormat: CVPixelBufferGetPixelFormatType(pixelBuffer)
            ),
            capture: Lip2DCaptureMetadata(
                source: "AVCaptureVideoDataOutput",
                cameraPosition: cameraPosition.rawValue,
                videoOrientation: "portrait",
                previewMirrored: cameraPosition.previewMirrored,
                savedBufferMirrored: false,
                sampleRateFPS: 5
            ),
            recordingGuide: Lip2DRecordingGuideFramePayload(
                phaseID: guideState.phase.id,
                phaseTitle: guideState.phase.title,
                elapsedSeconds: guideState.elapsedSeconds,
                totalSeconds: guideState.totalSeconds,
                phaseRemainingSeconds: guideState.phaseRemainingSeconds
            )
        )

        queue.async { [weak self] in
            self?.writeSample(payload: payload, pixelBuffer: pixelBuffer, sessionURL: sessionURL, sampleIndex: sampleIndex)
        }
    }

    private func writeSample(
        payload: Lip2DFrameMetadata,
        pixelBuffer: CVPixelBuffer,
        sessionURL: URL,
        sampleIndex: Int
    ) {
        do {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let options: [CIImageRepresentationOption: Any] = [
                CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.92
            ]
            guard let imageData = ciContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: options) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try imageData.write(to: sessionURL.appendingPathComponent(payload.image.file), options: .atomic)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let metadataData = try encoder.encode(payload)
            try metadataData.write(
                to: sessionURL.appendingPathComponent("frames").appendingPathComponent("\(payload.sampleID).json"),
                options: .atomic
            )

            if let line = String(data: metadataData, encoding: .utf8) {
                try append(line + "\n", to: sessionURL.appendingPathComponent("metadata.jsonl"))
            }

            lock.lock()
            completedSampleCount = max(completedSampleCount, sampleIndex + 1)
            let stillRecording = isRecording
            lock.unlock()

            if stillRecording {
                publishGuideStatus()
            }
        } catch {
            publishStatus("2D REC write failed: \(error.localizedDescription)")
        }
    }

    private func append(_ string: String, to url: URL) throws {
        guard let data = string.data(using: .utf8) else { return }
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.close()
        } else {
            try data.write(to: url, options: .atomic)
        }
    }

    private func startGuideTimer() {
        cancelGuideTimer()

        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now(), repeating: .milliseconds(250), leeway: .milliseconds(80))
        timer.setEventHandler { [weak self] in
            self?.updateGuideTimer()
        }
        guideTimer = timer
        timer.resume()
    }

    private func cancelGuideTimer() {
        guideTimer?.cancel()
        guideTimer = nil
    }

    private func updateGuideTimer() {
        let elapsedSeconds: Int
        let shouldFinish: Bool

        lock.lock()
        guard isRecording else {
            lock.unlock()
            cancelGuideTimer()
            return
        }
        elapsedSeconds = elapsedGuideSeconds(now: CACurrentMediaTime())
        shouldFinish = guide.state(atElapsedSeconds: elapsedSeconds).isFinished
        lock.unlock()

        publishGuideStatus(elapsedSeconds: elapsedSeconds)
        if shouldFinish {
            stop(completedAutomatically: true)
        }
    }

    private func publishGuideStatus(elapsedSeconds overrideElapsedSeconds: Int? = nil) {
        let elapsedSeconds: Int
        let count: Int

        lock.lock()
        guard isRecording else {
            lock.unlock()
            return
        }
        elapsedSeconds = overrideElapsedSeconds ?? elapsedGuideSeconds(now: CACurrentMediaTime())
        count = completedSampleCount
        lock.unlock()

        publishStatus(guide.statusText(elapsedSeconds: elapsedSeconds, sampleCount: count).replacingOccurrences(of: "REC", with: "2D REC"))
    }

    private func elapsedGuideSeconds(now: CFTimeInterval) -> Int {
        Int((now - recordingStartedAt).rounded(.down))
    }

    private func publishStatus(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onStatusChange?(text)
        }
    }

    private func publishRecordingActive(_ active: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.onRecordingActiveChange?(active)
        }
    }

    private static func recordingsRootURL() throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return documents.appendingPathComponent("Lip2DRecordings", isDirectory: true)
    }

    private static func sessionName() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }
}

private struct Lip2DSessionMetadata: Encodable {
    let schemaVersion: Int
    let createdAt: String
    let coordinateSystem: String
    let notes: [String]
}

private struct Lip2DFrameMetadata: Encodable {
    let schemaVersion: Int
    let sampleID: String
    let timestamp: TimeInterval
    let image: Lip2DImageMetadata
    let capture: Lip2DCaptureMetadata
    let recordingGuide: Lip2DRecordingGuideFramePayload?
}

private struct Lip2DImageMetadata: Encodable {
    let file: String
    let width: Int
    let height: Int
    let pixelFormat: OSType
}

private struct Lip2DCaptureMetadata: Encodable {
    let source: String
    let cameraPosition: String
    let videoOrientation: String
    let previewMirrored: Bool
    let savedBufferMirrored: Bool
    let sampleRateFPS: Int
}

private struct Lip2DRecordingGuideFramePayload: Encodable {
    let phaseID: String
    let phaseTitle: String
    let elapsedSeconds: Int
    let totalSeconds: Int
    let phaseRemainingSeconds: Int
}

private extension CGPoint {
    func applying(offset: CGPoint) -> CGPoint {
        CGPoint(x: x + offset.x, y: y + offset.y)
    }

    func offset(by direction: CGPoint, distance: CGFloat) -> CGPoint {
        CGPoint(
            x: x + direction.x * distance,
            y: y + direction.y * distance
        )
    }

    func interpolated(to point: CGPoint, amount: CGFloat) -> CGPoint {
        CGPoint(
            x: x + ((point.x - x) * amount),
            y: y + ((point.y - y) * amount)
        )
    }

    func distance(to point: CGPoint) -> CGFloat {
        hypot(x - point.x, y - point.y)
    }

    func limitedToward(_ point: CGPoint, maxDistance: CGFloat, deadzone: CGFloat = 0) -> CGPoint {
        let rawDelta = CGPoint(x: point.x - x, y: point.y - y)
        let distance = hypot(rawDelta.x, rawDelta.y)
        guard distance > deadzone else {
            return self
        }

        let adjustedDistance = max(distance - deadzone, 0)
        let adjustedDelta = rawDelta.scaled(by: adjustedDistance / max(distance, 0.001))
        let delta = adjustedDelta.limitedMagnitude(maxDistance)
        return applying(offset: delta)
    }

    func scaled(by amount: CGFloat) -> CGPoint {
        CGPoint(x: x * amount, y: y * amount)
    }

    func limitedMagnitude(_ maxMagnitude: CGFloat) -> CGPoint {
        let magnitude = hypot(x, y)
        guard magnitude > maxMagnitude, magnitude > 0.001 else {
            return self
        }

        let scale = maxMagnitude / magnitude
        return CGPoint(x: x * scale, y: y * scale)
    }
}
