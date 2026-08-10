//
//  ScreenLipMaskTracker.swift
//  arface
//
//  Created by Codex on 2026/6/6.
//

import CoreGraphics
import Foundation

struct ScreenLipMaskTracker {
    private(set) var lastMaskFrame: CGRect?
    private(set) var lastRegionOfInterest: CGRect?
    private(set) var lastDisplayFrame: CGRect?
    private(set) var lastUpdatedAt: CFTimeInterval?

    let maxTrackingAge: CFTimeInterval

    init(maxTrackingAge: CFTimeInterval = 0.70) {
        self.maxTrackingAge = maxTrackingAge
    }

    mutating func reset() {
        lastMaskFrame = nil
        lastRegionOfInterest = nil
        lastDisplayFrame = nil
        lastUpdatedAt = nil
    }

    @discardableResult
    mutating func acceptMaskFrame(
        _ maskFrame: CGRect,
        regionOfInterest: CGRect?,
        now: CFTimeInterval
    ) -> CGRect {
        let cleanMaskFrame = maskFrame.sanitizedForLipTracking ?? .zero
        let displayFrame = regionOfInterest?.sanitizedForLipTracking ?? cleanMaskFrame

        lastMaskFrame = cleanMaskFrame
        lastRegionOfInterest = displayFrame
        lastDisplayFrame = displayFrame
        lastUpdatedAt = now
        return displayFrame
    }

    mutating func trackedFrame(
        for regionOfInterest: CGRect,
        now: CFTimeInterval
    ) -> CGRect? {
        guard
            hasRecentMask(now: now),
            let lastRegionOfInterest,
            let lastDisplayFrame,
            let nextRegionOfInterest = regionOfInterest.sanitizedForLipTracking,
            isPlausibleMotion(from: lastRegionOfInterest, to: nextRegionOfInterest)
        else {
            return nil
        }

        let warpedFrame = warp(
            displayFrame: lastDisplayFrame,
            from: lastRegionOfInterest,
            to: nextRegionOfInterest
        )
        let smoothing = smoothingAmount(
            from: lastDisplayFrame,
            to: warpedFrame,
            previousROI: lastRegionOfInterest,
            nextROI: nextRegionOfInterest
        )
        let displayFrame = lastDisplayFrame.interpolated(to: warpedFrame, amount: smoothing)

        self.lastRegionOfInterest = nextRegionOfInterest
        self.lastDisplayFrame = displayFrame
        return displayFrame
    }

    func hasRecentMask(now: CFTimeInterval) -> Bool {
        guard let lastUpdatedAt else { return false }
        return now - lastUpdatedAt <= maxTrackingAge
    }

    private func warp(
        displayFrame: CGRect,
        from previousROI: CGRect,
        to nextROI: CGRect
    ) -> CGRect {
        let scaleX = nextROI.width / max(previousROI.width, 1)
        let scaleY = nextROI.height / max(previousROI.height, 1)
        let offsetX = (displayFrame.midX - previousROI.midX) * scaleX
        let offsetY = (displayFrame.midY - previousROI.midY) * scaleY
        let width = displayFrame.width * scaleX
        let height = displayFrame.height * scaleY
        return CGRect(
            x: nextROI.midX + offsetX - width * 0.5,
            y: nextROI.midY + offsetY - height * 0.5,
            width: width,
            height: height
        )
    }

    private func isPlausibleMotion(from previousROI: CGRect, to nextROI: CGRect) -> Bool {
        let scaleX = nextROI.width / max(previousROI.width, 1)
        let scaleY = nextROI.height / max(previousROI.height, 1)
        guard
            scaleX >= 0.35,
            scaleX <= 2.85,
            scaleY >= 0.35,
            scaleY <= 2.85
        else {
            return false
        }

        let maxTravel = max(previousROI.width, previousROI.height, nextROI.width, nextROI.height) * 1.15
        let travel = hypot(nextROI.midX - previousROI.midX, nextROI.midY - previousROI.midY)
        return travel <= maxTravel
    }

    private func smoothingAmount(
        from previousFrame: CGRect,
        to nextFrame: CGRect,
        previousROI: CGRect,
        nextROI: CGRect
    ) -> CGFloat {
        let scaleDelta = max(
            abs(nextROI.width / max(previousROI.width, 1) - 1),
            abs(nextROI.height / max(previousROI.height, 1) - 1)
        )
        let travel = hypot(nextFrame.midX - previousFrame.midX, nextFrame.midY - previousFrame.midY)
        let travelRatio = travel / max(max(nextROI.width, nextROI.height), 1)

        if scaleDelta > 0.06 {
            return 1.0
        }

        if travelRatio > 0.08 {
            return 0.92
        }

        return 0.78
    }
}

private extension CGRect {
    var sanitizedForLipTracking: CGRect? {
        let clean = standardized
        guard
            clean.origin.x.isFinite,
            clean.origin.y.isFinite,
            clean.width.isFinite,
            clean.height.isFinite,
            clean.width >= 4,
            clean.height >= 4
        else {
            return nil
        }

        return clean
    }

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
