//
//  ARKitUVSampleRecorder.swift
//  arface
//
//  Created by Codex on 2026/5/27.
//

import ARKit
import CoreImage
import Foundation
import ImageIO

final class ARKitUVSampleRecorder {
    var onStatusChange: ((String) -> Void)?
    var onRecordingActiveChange: ((Bool) -> Void)?

    private let queue = DispatchQueue(label: "com.perfectmakeup.arkit.uv-recorder", qos: .utility)
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

            let session = SessionMetadata(
                schemaVersion: 1,
                createdAt: ISO8601DateFormatter().string(from: Date()),
                coordinateSystem: "capturedImage pixels + ARKit face-anchor-local mesh + ARKit UV",
                notes: [
                    "image_jpeg is saved from ARFrame.capturedImage without display warping",
                    "display_transform_portrait maps captured image normalized coordinates to portrait viewport normalized coordinates",
                    "projection_matrix_portrait and view_matrix_portrait use UIInterfaceOrientation.portrait"
                ]
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let sessionData = try encoder.encode(session)
            try sessionData.write(to: sessionURL.appendingPathComponent("session.json"), options: .atomic)

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
            publishStatus("UV REC failed: \(error.localizedDescription)")
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

        publishStatus(completedAutomatically ? "UV REC 完成  saved \(count)" : "UV REC saved \(count)")
        publishRecordingActive(false)
    }

    func record(
        frame: ARFrame,
        faceAnchor: ARFaceAnchor,
        viewportSize: CGSize,
        faceROI: CGRect?,
        lipROI: CGRect?
    ) {
        let now = CACurrentMediaTime()

        lock.lock()
        guard isRecording, now - lastSampleTime >= minimumFrameInterval, let sessionURL else {
            lock.unlock()
            return
        }
        let sampleIndex = nextSampleIndex
        let guideElapsedSeconds = elapsedGuideSeconds(now: now)
        let guideState = guide.state(atElapsedSeconds: guideElapsedSeconds)
        nextSampleIndex += 1
        lastSampleTime = now
        lock.unlock()

        let sampleID = String(format: "frame_%06d", sampleIndex)
        let pixelBuffer = frame.capturedImage
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let displayTransform = frame.displayTransform(for: .portrait, viewportSize: viewportSize)

        let geometry = faceAnchor.geometry
        let payload = FrameMetadata(
            schemaVersion: 1,
            sampleID: sampleID,
            timestamp: frame.timestamp,
            image: ImageMetadata(
                file: "images/\(sampleID).jpg",
                width: imageWidth,
                height: imageHeight,
                pixelFormat: CVPixelBufferGetPixelFormatType(pixelBuffer)
            ),
            viewport: SizePayload(width: Double(viewportSize.width), height: Double(viewportSize.height)),
            displayTransformPortrait: displayTransform.payload,
            cameraImageResolution: SizePayload(
                width: Double(frame.camera.imageResolution.width),
                height: Double(frame.camera.imageResolution.height)
            ),
            cameraIntrinsics: frame.camera.intrinsics.rowMajorArray,
            cameraTransform: frame.camera.transform.rowMajorArray,
            projectionMatrixPortrait: frame.camera.projectionMatrix(
                for: .portrait,
                viewportSize: viewportSize,
                zNear: 0.001,
                zFar: 10
            ).rowMajorArray,
            viewMatrixPortrait: frame.camera.viewMatrix(for: .portrait).rowMajorArray,
            faceTransform: faceAnchor.transform.rowMajorArray,
            faceGeometry: FaceGeometryPayload(
                vertices: geometry.vertices.map { Vector3Payload($0) },
                textureCoordinates: geometry.textureCoordinates.map { Vector2Payload($0) },
                triangleIndices: geometry.triangleIndices.map(Int.init)
            ),
            blendShapes: faceAnchor.blendShapes.reduce(into: [String: Float]()) { result, element in
                result[element.key.rawValue] = element.value.floatValue
            },
            recordingGuide: RecordingGuideFramePayload(
                phaseID: guideState.phase.id,
                phaseTitle: guideState.phase.title,
                elapsedSeconds: guideState.elapsedSeconds,
                totalSeconds: guideState.totalSeconds,
                phaseRemainingSeconds: guideState.phaseRemainingSeconds
            ),
            faceROI: faceROI.map(RectPayload.init),
            lipROI: lipROI.map(RectPayload.init)
        )

        queue.async { [weak self] in
            self?.writeSample(payload: payload, pixelBuffer: pixelBuffer, sessionURL: sessionURL, sampleIndex: sampleIndex)
        }
    }

    private func writeSample(
        payload: FrameMetadata,
        pixelBuffer: CVPixelBuffer,
        sessionURL: URL,
        sampleIndex: Int
    ) {
        do {
            let imageURL = sessionURL.appendingPathComponent(payload.image.file)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let options: [CIImageRepresentationOption: Any] = [
                CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.88
            ]
            guard let imageData = ciContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: options) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try imageData.write(to: imageURL, options: .atomic)

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
            publishStatus("UV REC write failed: \(error.localizedDescription)")
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

        publishStatus(guide.statusText(elapsedSeconds: elapsedSeconds, sampleCount: count))
    }

    private func elapsedGuideSeconds(now: CFTimeInterval) -> Int {
        Int((now - recordingStartedAt).rounded(.down))
    }

    private static func recordingsRootURL() throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return documents.appendingPathComponent("ARKitLipUVRecordings", isDirectory: true)
    }

    private static func sessionName() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }
}

private struct SessionMetadata: Encodable {
    let schemaVersion: Int
    let createdAt: String
    let coordinateSystem: String
    let notes: [String]
}

private struct FrameMetadata: Encodable {
    let schemaVersion: Int
    let sampleID: String
    let timestamp: TimeInterval
    let image: ImageMetadata
    let viewport: SizePayload
    let displayTransformPortrait: AffineTransformPayload
    let cameraImageResolution: SizePayload
    let cameraIntrinsics: [Float]
    let cameraTransform: [Float]
    let projectionMatrixPortrait: [Float]
    let viewMatrixPortrait: [Float]
    let faceTransform: [Float]
    let faceGeometry: FaceGeometryPayload
    let blendShapes: [String: Float]
    let recordingGuide: RecordingGuideFramePayload?
    let faceROI: RectPayload?
    let lipROI: RectPayload?
}

private struct ImageMetadata: Encodable {
    let file: String
    let width: Int
    let height: Int
    let pixelFormat: OSType
}

private struct FaceGeometryPayload: Encodable {
    let vertices: [Vector3Payload]
    let textureCoordinates: [Vector2Payload]
    let triangleIndices: [Int]
}

private struct RecordingGuideFramePayload: Encodable {
    let phaseID: String
    let phaseTitle: String
    let elapsedSeconds: Int
    let totalSeconds: Int
    let phaseRemainingSeconds: Int
}

private struct Vector2Payload: Encodable {
    let x: Float
    let y: Float

    init(_ vector: simd_float2) {
        x = vector.x
        y = vector.y
    }
}

private struct Vector3Payload: Encodable {
    let x: Float
    let y: Float
    let z: Float

    init(_ vector: simd_float3) {
        x = vector.x
        y = vector.y
        z = vector.z
    }
}

private struct SizePayload: Encodable {
    let width: Double
    let height: Double
}

private struct RectPayload: Encodable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    nonisolated init(_ rect: CGRect) {
        x = Double(rect.origin.x)
        y = Double(rect.origin.y)
        width = Double(rect.width)
        height = Double(rect.height)
    }
}

private struct AffineTransformPayload: Encodable {
    let a: Double
    let b: Double
    let c: Double
    let d: Double
    let tx: Double
    let ty: Double
}

private extension CGAffineTransform {
    var payload: AffineTransformPayload {
        AffineTransformPayload(
            a: Double(a),
            b: Double(b),
            c: Double(c),
            d: Double(d),
            tx: Double(tx),
            ty: Double(ty)
        )
    }
}

private extension simd_float3x3 {
    var rowMajorArray: [Float] {
        [
            columns.0.x, columns.1.x, columns.2.x,
            columns.0.y, columns.1.y, columns.2.y,
            columns.0.z, columns.1.z, columns.2.z
        ]
    }
}

private extension simd_float4x4 {
    var rowMajorArray: [Float] {
        [
            columns.0.x, columns.1.x, columns.2.x, columns.3.x,
            columns.0.y, columns.1.y, columns.2.y, columns.3.y,
            columns.0.z, columns.1.z, columns.2.z, columns.3.z,
            columns.0.w, columns.1.w, columns.2.w, columns.3.w
        ]
    }
}
