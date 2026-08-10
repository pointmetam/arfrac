//
//  MakeupLook.swift
//  arface
//
//  Created by lee on 2026/5/21.
//

import Combine
import SwiftUI

enum MakeupTrackingStatus: Equatable {
    case checking
    case unsupported
    case ready
    case tracking
    case limited(String)
    case interrupted
    case failed(String)

    var title: String {
        switch self {
        case .checking:
            "正在检查相机"
        case .unsupported:
            "此设备不支持面部 AR"
        case .ready:
            "相机已就绪"
        case .tracking:
            "正在试妆"
        case .limited(let reason):
            reason
        case .interrupted:
            "相机会话已暂停"
        case .failed(let message):
            message
        }
    }

    var tint: Color {
        switch self {
        case .tracking, .ready:
            .mint
        case .checking, .limited:
            .yellow
        case .unsupported, .interrupted, .failed:
            .orange
        }
    }
}

struct MakeupLook: Identifiable, Hashable {
    let id: String
    let shortName: String
    let name: String
    let foundation: RGBAColor
    let blush: RGBAColor
    let lip: RGBAColor
    let eye: RGBAColor

    var accentColor: Color {
        lip.color
    }

    var swatch: LinearGradient {
        LinearGradient(
            colors: [lip.color.opacity(0.78), lip.color, lip.color.opacity(0.92)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    var lipLineDebugSource: LipLineDebugSource {
        switch id {
        case "debug-lip-line":
            .fused
        case "debug-lip-line-uv":
            .uvOnly
        case "debug-lip-line-color":
            .colorOnly
        default:
            .none
        }
    }

    var isLipLineDebug: Bool {
        lipLineDebugSource != .none
    }

    var usesLipLineColorFusion: Bool {
        lipLineDebugSource == .fused
    }
}

enum LipLineDebugSource: Equatable {
    case none
    case fused
    case uvOnly
    case colorOnly
}

enum Lip2DCameraPosition: String, Equatable {
    case front
    case back

    var displayName: String {
        switch self {
        case .front:
            "前摄"
        case .back:
            "后摄"
        }
    }

    var previewMirrored: Bool {
        self == .front
    }
}

struct RGBAColor: Hashable {
    let red: CGFloat
    let green: CGFloat
    let blue: CGFloat
    let alpha: CGFloat

    var color: Color {
        Color(red: red, green: green, blue: blue, opacity: alpha)
    }

    func uiColor(intensity: Double) -> UIColor {
        UIColor(
            red: red,
            green: green,
            blue: blue,
            alpha: alpha * CGFloat(intensity)
        )
    }
}

enum LipMakeupMode: String, CaseIterable, Identifiable {
    case autoFusion
    case arkit3D
    case faceParsing
    case lipRefinement
    case lipRefinementLarge
    case lipRefinementLarge3DGPU
    case lipSDFRefinement
    case arkitUVStudent
    case arkitUVSegStudent

    // Curve refinement is now folded into UV-4C instead of being exposed as a separate mode.
    static let visibleModes: [LipMakeupMode] = [.arkitUVSegStudent, .lipSDFRefinement, .lipRefinement]

    var id: String { rawValue }

    var title: String {
        switch self {
        case .autoFusion:
            "自动融合"
        case .arkit3D:
            "3D跟随"
        case .faceParsing:
            "全脸分区"
        case .lipRefinement:
            "候选模型"
        case .lipRefinementLarge:
            "唇线B大"
        case .lipRefinementLarge3DGPU:
            "B+3D混合"
        case .lipSDFRefinement:
            "SDF唇线"
        case .arkitUVStudent:
            "UV模型"
        case .arkitUVSegStudent:
            "UV分区"
        }
    }

    var pickerTitle: String {
        switch self {
        case .autoFusion:
            "自动"
        case .arkit3D:
            "3D"
        case .faceParsing:
            "全脸"
        case .lipRefinement:
            "候选"
        case .lipRefinementLarge:
            "B大"
        case .lipRefinementLarge3DGPU:
            "B3D"
        case .lipSDFRefinement:
            "SDF"
        case .arkitUVStudent:
            "UV-A"
        case .arkitUVSegStudent:
            "UV-4C"
        }
    }

    var usesModel: Bool {
        self == .autoFusion
            || self == .faceParsing
            || self == .lipRefinement
            || self == .lipRefinementLarge
            || self == .lipRefinementLarge3DGPU
            || self == .lipSDFRefinement
            || usesUVLipStudent
    }

    var isLipRefinement: Bool {
        self == .lipRefinement || self == .lipRefinementLarge || self == .lipRefinementLarge3DGPU || self == .lipSDFRefinement
    }

    var usesLipROIModel: Bool {
        self == .lipRefinement || self == .lipRefinementLarge || self == .lipRefinementLarge3DGPU || self == .lipSDFRefinement
    }

    var usesLargeLipModel: Bool {
        self == .lipRefinementLarge || self == .lipRefinementLarge3DGPU
    }

    var usesUVLipStudent: Bool {
        self == .arkitUVStudent || self == .arkitUVSegStudent
    }

    var usesUVSegStudent: Bool {
        self == .arkitUVSegStudent
    }

    var uses3DLipBase: Bool {
        self == .arkit3D || self == .lipRefinementLarge3DGPU || usesUVLipStudent
    }

    func usesStartup3DLipFallback(hasUVLipAlphaMask: Bool, isLipLineDebug: Bool) -> Bool {
        usesUVLipStudent && !hasUVLipAlphaMask && !isLipLineDebug
    }

    var showsScreenModelOverlay: Bool {
        self == .faceParsing
            || self == .lipRefinement
            || self == .lipRefinementLarge
            || self == .lipRefinementLarge3DGPU
    }

    var usesGPULipOverlay: Bool {
        false
    }

    var usesUVProjectedLipRefinement: Bool {
        self == .lipSDFRefinement
    }

    var modelOverlayOpacity: Float {
        switch self {
        case .lipRefinement:
            0.84
        case .lipRefinementLarge:
            0.96
        case .lipRefinementLarge3DGPU:
            0.76
        case .lipSDFRefinement:
            0.96
        case .autoFusion, .arkit3D, .faceParsing, .arkitUVStudent, .arkitUVSegStudent:
            0.94
        }
    }

    var performanceLabel: String? {
        switch self {
        case .autoFusion:
            "Auto"
        case .arkit3D:
            nil
        case .faceParsing:
            "Face ML"
        case .lipRefinement:
            "Lip ML 候选"
        case .lipRefinementLarge:
            "Lip ML B"
        case .lipRefinementLarge3DGPU:
            "Lip ML B+3D"
        case .lipSDFRefinement:
            "Lip SDF"
        case .arkitUVStudent:
            "UV ML"
        case .arkitUVSegStudent:
            "UV 4C"
        }
    }
}

@MainActor
final class MakeupState: ObservableObject {
    let looks: [MakeupLook] = [
        MakeupLook(
            id: "lancome-211",
            shortName: "#211",
            name: "#211 纯粹裸",
            foundation: RGBAColor(red: 0.89, green: 0.56, blue: 0.55, alpha: 0.12),
            blush: RGBAColor(red: 0.79, green: 0.37, blue: 0.42, alpha: 0.24),
            lip: RGBAColor(red: 0.86, green: 0.24, blue: 0.32, alpha: 0.95),
            eye: RGBAColor(red: 0.58, green: 0.28, blue: 0.31, alpha: 0.20)
        ),
        MakeupLook(
            id: "lancome-274",
            shortName: "#274",
            name: "#274 奶茶裸",
            foundation: RGBAColor(red: 0.92, green: 0.58, blue: 0.52, alpha: 0.12),
            blush: RGBAColor(red: 0.83, green: 0.39, blue: 0.36, alpha: 0.26),
            lip: RGBAColor(red: 0.88, green: 0.27, blue: 0.24, alpha: 0.95),
            eye: RGBAColor(red: 0.60, green: 0.29, blue: 0.27, alpha: 0.20)
        ),
        MakeupLook(
            id: "lancome-212",
            shortName: "#212",
            name: "#212 智感裸",
            foundation: RGBAColor(red: 0.82, green: 0.48, blue: 0.45, alpha: 0.12),
            blush: RGBAColor(red: 0.69, green: 0.31, blue: 0.30, alpha: 0.24),
            lip: RGBAColor(red: 0.73, green: 0.20, blue: 0.21, alpha: 0.95),
            eye: RGBAColor(red: 0.48, green: 0.24, blue: 0.23, alpha: 0.20)
        ),
        MakeupLook(
            id: "lancome-242",
            shortName: "#242",
            name: "#242 本色裸",
            foundation: RGBAColor(red: 0.84, green: 0.34, blue: 0.29, alpha: 0.12),
            blush: RGBAColor(red: 0.67, green: 0.17, blue: 0.14, alpha: 0.30),
            lip: RGBAColor(red: 0.66, green: 0.09, blue: 0.07, alpha: 0.96),
            eye: RGBAColor(red: 0.42, green: 0.12, blue: 0.10, alpha: 0.22)
        ),
        MakeupLook(
            id: "edge-check-green",
            shortName: "G",
            name: "G 唇线绿",
            foundation: RGBAColor(red: 0.60, green: 0.72, blue: 0.62, alpha: 0.02),
            blush: RGBAColor(red: 0.22, green: 0.72, blue: 0.36, alpha: 0.02),
            lip: RGBAColor(red: 0.20, green: 0.94, blue: 0.08, alpha: 0.96),
            eye: RGBAColor(red: 0.12, green: 0.44, blue: 0.16, alpha: 0.02)
        ),
        MakeupLook(
            id: "debug-green-lip",
            shortName: "DBG-G",
            name: "DBG-G 覆盖检查",
            foundation: RGBAColor(red: 0.60, green: 0.72, blue: 0.62, alpha: 0.02),
            blush: RGBAColor(red: 0.22, green: 0.72, blue: 0.36, alpha: 0.02),
            lip: RGBAColor(red: 0.02, green: 0.95, blue: 0.18, alpha: 0.96),
            eye: RGBAColor(red: 0.12, green: 0.44, blue: 0.16, alpha: 0.02)
        ),
        MakeupLook(
            id: "debug-lip-line",
            shortName: "LINE",
            name: "LINE 融合唇线",
            foundation: RGBAColor(red: 0.60, green: 0.72, blue: 0.62, alpha: 0.00),
            blush: RGBAColor(red: 0.22, green: 0.72, blue: 0.36, alpha: 0.00),
            lip: RGBAColor(red: 0.00, green: 1.00, blue: 0.18, alpha: 0.96),
            eye: RGBAColor(red: 0.12, green: 0.44, blue: 0.16, alpha: 0.00)
        ),
        MakeupLook(
            id: "debug-lip-line-color",
            shortName: "COLOR",
            name: "COLOR 纯颜色唇线",
            foundation: RGBAColor(red: 0.60, green: 0.72, blue: 0.62, alpha: 0.00),
            blush: RGBAColor(red: 0.22, green: 0.72, blue: 0.36, alpha: 0.00),
            lip: RGBAColor(red: 0.00, green: 1.00, blue: 0.18, alpha: 0.96),
            eye: RGBAColor(red: 0.12, green: 0.44, blue: 0.16, alpha: 0.00)
        ),
        MakeupLook(
            id: "debug-lip-line-uv",
            shortName: "UV",
            name: "UV 纯UV唇线",
            foundation: RGBAColor(red: 0.45, green: 0.70, blue: 0.86, alpha: 0.00),
            blush: RGBAColor(red: 0.16, green: 0.54, blue: 0.84, alpha: 0.00),
            lip: RGBAColor(red: 0.00, green: 0.86, blue: 1.00, alpha: 0.96),
            eye: RGBAColor(red: 0.10, green: 0.34, blue: 0.54, alpha: 0.00)
        )
    ]

    @Published var selectedLook: MakeupLook
    @Published var intensity = 0.88
    @Published var isMakeupVisible = true
    @Published var lipMode: LipMakeupMode = .lipSDFRefinement
    @Published var showFaceMesh = false
    @Published var isLipOnlyTestMode = true
    @Published var forceRawUVOnly = false
    @Published var isRecordingARKitUV = false
    @Published var isRecordingLip2D = false
    @Published var lip2DCameraPosition: Lip2DCameraPosition = .back
    @Published var photoCaptureRequestID = 0
    @Published var status: MakeupTrackingStatus = .checking
    @Published var performanceText = ""
    @Published var recordingText = ""
    @Published var photoText = ""

    var visibleLooks: [MakeupLook] {
        looks.filter { !$0.isLipLineDebug }
    }

    var isTrainingRecordingActive: Bool {
        isRecordingARKitUV || isRecordingLip2D
    }

    var usesLip2DCaptureView: Bool {
        isRecordingLip2D
    }

    var allowsLiveLipDetection: Bool {
        !isRecordingLip2D
    }

    var trainingRecordingAccessibilityLabel: String {
        if isRecordingLip2D {
            return "提前停止高清 2D 唇部录制"
        }
        if isRecordingARKitUV {
            return "提前停止 ARKit UV 引导录制"
        }
        return "开始高清 2D 唇部录制"
    }

    var isMakeupVisibleForARKit: Bool {
        isMakeupVisible && !isTrainingRecordingActive
    }

    func markLip2DRecordingButtonTapped() {
        do {
            let documents = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            let root = documents.appendingPathComponent("Lip2DRecordings", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

            let payload = """
            {
              "event": "lip2d_record_button_tapped",
              "createdAt": "\(ISO8601DateFormatter().string(from: Date()))"
            }
            """
            try payload.write(
                to: root.appendingPathComponent("_last_button_tap.json"),
                atomically: true,
                encoding: .utf8
            )
        } catch {
            recordingText = "2D REC marker failed: \(error.localizedDescription)"
        }
    }

    init() {
        selectedLook = looks[1]
    }
}
