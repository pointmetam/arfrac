//
//  AutoLipRenderDecision.swift
//  arface
//
//  Created by Codex on 2026/6/6.
//

import CoreGraphics
import Foundation

enum AutoLipRenderPath: Equatable {
    case uv4c
    case screen2D
    case freeze
    case none

    var usesUVAlphaMask: Bool {
        self == .uv4c
    }

    var usesScreenOverlay: Bool {
        self == .screen2D || self == .freeze
    }

    var displayName: String {
        switch self {
        case .uv4c:
            "UV-4C"
        case .screen2D:
            "2D补位"
        case .freeze:
            "Freeze"
        case .none:
            "None"
        }
    }
}

struct AutoLipRenderDecider {
    static let closeUpMouthROIRatio: CGFloat = 0.58

    static func decide(
        mouthROIRatio: CGFloat?,
        uvMaskCoverage: CGFloat?,
        uvMaskMaximumAlpha: Int?,
        hasFreshScreenMask: Bool,
        hasRecentScreenMask: Bool,
        uvModelReady: Bool,
        screenModelReady: Bool
    ) -> AutoLipRenderPath {
        guard uvModelReady || screenModelReady else { return .none }

        let isCloseUp = mouthROIRatio.map { $0 >= closeUpMouthROIRatio } ?? false

        if screenModelReady, isCloseUp {
            return fallbackScreenPath(hasFreshScreenMask: hasFreshScreenMask, hasRecentScreenMask: hasRecentScreenMask)
        }

        if uvModelReady {
            return .uv4c
        }

        if screenModelReady {
            return fallbackScreenPath(hasFreshScreenMask: hasFreshScreenMask, hasRecentScreenMask: hasRecentScreenMask)
        }

        return .none
    }

    private static func fallbackScreenPath(hasFreshScreenMask: Bool, hasRecentScreenMask: Bool) -> AutoLipRenderPath {
        if hasFreshScreenMask {
            return .screen2D
        }

        if hasRecentScreenMask {
            return .freeze
        }

        return .screen2D
    }
}
