//
//  UVMakeupTexture.swift
//  arface
//
//  Created by lee on 2026/5/21.
//

import ARKit
import Metal
import UIKit

struct UVLipAlphaMask: @unchecked Sendable {
    let width: Int
    let height: Int
    let alpha: [UInt8]

    private struct ProductRenderCacheEntry {
        let signature: UInt64
        let mask: UVLipAlphaMask
    }

    nonisolated(unsafe) private static var productRenderCache: ProductRenderCacheEntry?
    private static let productRenderCacheLock = NSLock()

    var nonzeroCoverage: CGFloat {
        guard !alpha.isEmpty else { return 0 }
        let nonzeroPixels = alpha.reduce(0) { count, value in
            count + (value > 0 ? 1 : 0)
        }
        return CGFloat(nonzeroPixels) / CGFloat(alpha.count)
    }

    var maximumAlpha: UInt8 {
        alpha.max() ?? 0
    }

    private func productRenderSignature(targetSize: Int, preserveClosedMouthCrease: Bool) -> UInt64 {
        var hash: UInt64 = 1469598103934665603
        func mix(_ value: UInt64) {
            hash ^= value
            hash &*= 1099511628211
        }

        mix(UInt64(width))
        mix(UInt64(height))
        mix(UInt64(targetSize))
        mix(preserveClosedMouthCrease ? 1 : 0)
        for value in alpha {
            mix(UInt64(value))
        }
        return hash
    }

    func translated(xPixels: Int, yPixels: Int) -> UVLipAlphaMask {
        guard width > 0, height > 0, alpha.count == width * height else { return self }
        guard xPixels != 0 || yPixels != 0 else { return self }

        var output = [UInt8](repeating: 0, count: alpha.count)
        for y in 0..<height {
            let sourceY = y - yPixels
            guard sourceY >= 0, sourceY < height else { continue }

            for x in 0..<width {
                let sourceX = x - xPixels
                guard sourceX >= 0, sourceX < width else { continue }
                output[y * width + x] = alpha[sourceY * width + sourceX]
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func contracted(
        threshold: UInt8 = 24,
        radius: Int = 1,
        minimumNeighborRatio: CGFloat = 0.72
    ) -> UVLipAlphaMask {
        guard radius > 0, width > 2, height > 2, alpha.count == width * height else { return self }

        let requiredRatio = min(max(minimumNeighborRatio, 0), 1)
        var output = [UInt8](repeating: 0, count: alpha.count)
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let value = alpha[index]
                guard value > threshold else { continue }

                var active = 0
                var total = 0
                var peak = Int(value)
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                        let sample = Int(alpha[sampleY * width + sampleX])
                        if sample > Int(threshold) {
                            active += 1
                            peak = max(peak, sample)
                        }
                        total += 1
                    }
                }

                let ratio = CGFloat(active) / CGFloat(max(total, 1))
                if ratio >= requiredRatio {
                    output[index] = UInt8(min(max(peak, 0), 255))
                }
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func temporallySmoothed(previous: UVLipAlphaMask?, response: CGFloat) -> UVLipAlphaMask {
        guard
            let previous,
            previous.width == width,
            previous.height == height,
            previous.alpha.count == alpha.count
        else {
            return self
        }

        let amount = min(max(response, 0), 1)
        var output = [UInt8](repeating: 0, count: alpha.count)
        for index in alpha.indices {
            let oldValue = CGFloat(previous.alpha[index])
            let newValue = CGFloat(alpha[index])
            output[index] = UInt8(min(max(oldValue + ((newValue - oldValue) * amount), 0), 255))
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func temporallyStabilized(previous: UVLipAlphaMask?, response: CGFloat, decay: CGFloat) -> UVLipAlphaMask {
        guard
            let previous,
            previous.width == width,
            previous.height == height,
            previous.alpha.count == alpha.count
        else {
            return self
        }

        let amount = min(max(response, 0), 1)
        let hold = min(max(decay, 0), 1)
        var output = [UInt8](repeating: 0, count: alpha.count)
        for index in alpha.indices {
            let oldValue = CGFloat(previous.alpha[index])
            let newValue = CGFloat(alpha[index])
            let blended = oldValue + ((newValue - oldValue) * amount)
            let retained = oldValue * hold
            output[index] = UInt8(min(max(max(blended, retained), 0), 255))
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func preservedClosedMouthCrease(
        threshold: UInt8 = 28,
        centerValleyThreshold: UInt8 = 196,
        minimumDepth: Int = 16
    ) -> UVLipAlphaMask {
        guard width > 8, height > 8, alpha.count == width * height else { return self }

        var topByColumn = [Int](repeating: Int.max, count: width)
        var bottomByColumn = [Int](repeating: -1, count: width)
        var minY = Int.max
        var maxY = -1
        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width where alpha[rowOffset + x] > threshold {
                topByColumn[x] = min(topByColumn[x], y)
                bottomByColumn[x] = max(bottomByColumn[x], y)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }

        let activeColumns = bottomByColumn.indices.filter { bottomByColumn[$0] >= 0 }
        guard
            let minX = activeColumns.first,
            let maxX = activeColumns.last,
            maxX - minX >= 10,
            maxY > minY
        else {
            return self
        }

        let lipHeight = maxY - minY + 1
        let lipWidth = maxX - minX + 1
        let centerX = CGFloat(minX + maxX) * 0.5
        let halfWidth = max(CGFloat(lipWidth) * 0.5, 1)
        let compactLip = CGFloat(lipHeight) / CGFloat(max(lipWidth, 1)) < 0.58

        var existingGapColumns = 0
        for x in minX...maxX {
            let top = topByColumn[x]
            let bottom = bottomByColumn[x]
            guard bottom - top >= 5 else { continue }

            let searchTop = top + max(1, (bottom - top) / 4)
            let searchBottom = bottom - max(1, (bottom - top) / 4)
            guard searchBottom > searchTop else { continue }

            let hasGap = (searchTop...searchBottom).contains { y in
                alpha[y * width + x] <= threshold
            }
            if hasGap {
                existingGapColumns += 1
            }
        }
        if existingGapColumns >= max(6, lipWidth / 5) {
            return self
        }

        var output = alpha
        var cutColumns = 0
        for x in minX...maxX {
            let top = topByColumn[x]
            let bottom = bottomByColumn[x]
            let span = bottom - top + 1
            guard span >= 7 else { continue }

            let searchTop = top + max(1, span / 4)
            let searchBottom = bottom - max(1, span / 4)
            guard searchBottom > searchTop else { continue }

            var valleyY = searchTop
            var valleyAlpha = Int.max
            for y in searchTop...searchBottom {
                let value = Int(alpha[y * width + x])
                if value < valleyAlpha {
                    valleyAlpha = value
                    valleyY = y
                }
            }

            let upperShoulderY = max(top, valleyY - 1)
            let lowerShoulderY = min(bottom, valleyY + 1)
            let shoulderAlpha = max(
                Int(alpha[upperShoulderY * width + x]),
                Int(alpha[lowerShoulderY * width + x])
            )
            let valleyDepth = shoulderAlpha - valleyAlpha
            let centerProgress = 1 - min(abs(CGFloat(x) - centerX) / halfWidth, 1)
            let centerWeight = Self.smoothstep(centerProgress)
            let isFlatBlock = valleyDepth < 8 && valleyAlpha > 180
            let shouldCut = valleyDepth >= minimumDepth
                || (compactLip && centerWeight > 0.22 && valleyAlpha < Int(centerValleyThreshold))
                || (existingGapColumns < 3 && isFlatBlock && centerWeight > 0.30)
            guard shouldCut else { continue }

            let targetRow: Int
            if isFlatBlock {
                targetRow = (top + bottom) / 2
            } else {
                targetRow = valleyY
            }

            func clearRow(_ row: Int) {
                guard row >= 0, row < height else { return }
                output[row * width + x] = 0
            }

            let gapHalfWidth: Int
            if centerWeight > 0.90 {
                gapHalfWidth = 2
            } else if centerWeight > 0.58 {
                gapHalfWidth = 1
            } else {
                gapHalfWidth = 0
            }

            for offset in -gapHalfWidth...gapHalfWidth {
                clearRow(targetRow + offset)
            }

            cutColumns += 1
        }

        guard cutColumns >= max(5, lipWidth / 6) else { return self }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func cgImage() -> CGImage? {
        guard width > 0, height > 0, alpha.count == width * height else { return nil }

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for index in alpha.indices {
            let pixelOffset = index * 4
            pixels[pixelOffset] = 255
            pixels[pixelOffset + 1] = 255
            pixels[pixelOffset + 2] = 255
            pixels[pixelOffset + 3] = alpha[index]
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

    func lowerContourClampedForLipLine(
        threshold: UInt8 = 40,
        centerInsetRatio: CGFloat = 0.12,
        sideInsetRatio: CGFloat = 0.025,
        fadePixels: Int = 2
    ) -> UVLipAlphaMask {
        guard width > 8, height > 8, alpha.count == width * height else { return self }

        var topByColumn = [Int](repeating: Int.max, count: width)
        var bottomByColumn = [Int](repeating: -1, count: width)
        var minY = Int.max
        var maxY = -1
        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width where alpha[rowOffset + x] > threshold {
                topByColumn[x] = min(topByColumn[x], y)
                bottomByColumn[x] = y
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }

        let activeColumns = bottomByColumn.indices.filter { bottomByColumn[$0] >= 0 }
        guard let minX = activeColumns.first, let maxX = activeColumns.last, maxX - minX > 6, maxY > minY else {
            return self
        }

        let activeWidth = maxX - minX + 1
        let lipHeight = maxY - minY + 1
        let medianRadius = max(6, min(18, activeWidth / 4))
        let allowedDrop = max(1, Int(round(CGFloat(height) * 0.008)))
        let centerInsetPixels = max(2, Int(round(CGFloat(lipHeight) * centerInsetRatio)))
        let sideInsetPixels = max(0, Int(round(CGFloat(lipHeight) * sideInsetRatio)))
        let centerX = CGFloat(minX + maxX) * 0.5
        let halfWidth = max(CGFloat(activeWidth) * 0.5, 1)
        let fade = max(fadePixels, 1)
        var output = alpha

        for x in minX...maxX {
            let bottom = bottomByColumn[x]
            guard bottom >= 0 else { continue }

            let windowStart = max(minX, x - medianRadius)
            let windowEnd = min(maxX, x + medianRadius)
            let localBottoms = (windowStart...windowEnd)
                .map { bottomByColumn[$0] }
                .filter { $0 >= 0 }
            guard !localBottoms.isEmpty else { continue }

            let localMedian = Self.median(localBottoms)
            let distanceFromCenter = min(abs(CGFloat(x) - centerX) / halfWidth, 1)
            let centerWeight = Self.smoothstep(1 - distanceFromCenter)
            let shapeInset = sideInsetPixels + Int(round(CGFloat(centerInsetPixels - sideInsetPixels) * centerWeight))
            let globalLimit = bottom - shapeInset
            let localLimit = localMedian + allowedDrop
            let contourLimit = max(topByColumn[x] + 4, min(globalLimit, localLimit))
            guard bottom > contourLimit else { continue }

            for y in (contourLimit + 1)..<height {
                let index = y * width + x
                let distance = y - contourLimit
                if distance >= fade {
                    output[index] = 0
                } else {
                    let factor = CGFloat(fade - distance) / CGFloat(fade)
                    output[index] = UInt8(min(max(CGFloat(output[index]) * factor, 0), 255))
                }
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func productLipSilhouetteRefined(
        threshold: UInt8 = 36,
        upperCenterInsetRatio: CGFloat = 0.025,
        upperCenterWidthRatio: CGFloat = 0.10,
        lowerCenterInsetRatio: CGFloat = 0.0,
        lowerSideInsetRatio: CGFloat = 0.08
    ) -> UVLipAlphaMask {
        guard width > 8, height > 8, alpha.count == width * height else { return self }

        var topByColumn = [Int](repeating: Int.max, count: width)
        var bottomByColumn = [Int](repeating: -1, count: width)
        var minY = Int.max
        var maxY = -1
        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width where alpha[rowOffset + x] > threshold {
                topByColumn[x] = min(topByColumn[x], y)
                bottomByColumn[x] = max(bottomByColumn[x], y)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }

        let activeColumns = bottomByColumn.indices.filter { bottomByColumn[$0] >= 0 }
        guard let minX = activeColumns.first, let maxX = activeColumns.last, maxX - minX > 6, maxY > minY else {
            return self
        }

        let lipHeight = maxY - minY + 1
        let centerX = CGFloat(minX + maxX) * 0.5
        let halfWidth = max(CGFloat(maxX - minX + 1) * 0.5, 1)
        let upperMaxInset = max(0, Int(round(CGFloat(lipHeight) * upperCenterInsetRatio)))
        let lowerCenterInset = max(0, Int(round(CGFloat(lipHeight) * lowerCenterInsetRatio)))
        let lowerSideInset = max(lowerCenterInset, Int(round(CGFloat(lipHeight) * lowerSideInsetRatio)))
        var output = alpha

        for x in minX...maxX {
            let top = topByColumn[x]
            let bottom = bottomByColumn[x]
            guard top < Int.max, bottom >= 0, bottom > top else { continue }

            let distanceFromCenter = min(abs(CGFloat(x) - centerX) / halfWidth, 1)
            let upperWeight = Self.smoothstep(1 - min(distanceFromCenter / max(upperCenterWidthRatio, 0.001), 1))
            let upperLimit = min(bottom, top + Int(round(CGFloat(upperMaxInset) * upperWeight)))
            if upperLimit > top {
                for y in top..<upperLimit {
                    output[y * width + x] = 0
                }
            }

            let sideWeight = Self.smoothstep(distanceFromCenter)
            let lowerInset = lowerCenterInset + Int(round(CGFloat(lowerSideInset - lowerCenterInset) * sideWeight))
            let lowerLimit = max(top, bottom - lowerInset)
            if bottom > lowerLimit {
                for y in (lowerLimit + 1)...bottom {
                    output[y * width + x] = 0
                }
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func softlyGated(
        by projectedLipLineMask: UVLipAlphaMask,
        minimumOverlapRatio: CGFloat = 0.24,
        outsideRetain: CGFloat = 0.14
    ) -> (mask: UVLipAlphaMask, applied: Bool, overlapRatio: CGFloat) {
        guard
            width == projectedLipLineMask.width,
            height == projectedLipLineMask.height,
            alpha.count == width * height,
            projectedLipLineMask.alpha.count == alpha.count
        else {
            return (self, false, 0)
        }

        let baseThreshold: UInt8 = 32
        let gateThreshold: UInt8 = 20
        var baseActive = 0
        var overlap = 0
        for index in alpha.indices where alpha[index] > baseThreshold {
            baseActive += 1
            if projectedLipLineMask.alpha[index] > gateThreshold {
                overlap += 1
            }
        }

        guard baseActive > 32 else { return (self, false, 0) }

        let overlapRatio = CGFloat(overlap) / CGFloat(baseActive)
        guard overlapRatio >= minimumOverlapRatio else {
            return (self, false, overlapRatio)
        }

        let expandedGate = Self.softlyExpanded(
            alpha: projectedLipLineMask.alpha,
            width: width,
            height: height,
            radius: 2
        )
        let softGate = Self.blurred(alpha: expandedGate, width: width, height: height, radius: 1)
        let retain = min(max(outsideRetain, 0), 0.75)
        var output = alpha

        for index in alpha.indices {
            let baseAlpha = CGFloat(alpha[index])
            guard baseAlpha > 0 else { continue }

            let gate = Self.smoothstep(CGFloat(softGate[index]) / 255)
            let factor = retain + ((1 - retain) * gate)
            output[index] = UInt8(min(max(baseAlpha * factor, 0), 255))
        }

        return (UVLipAlphaMask(width: width, height: height, alpha: output), true, overlapRatio)
    }

    func filledByLipLine(
        _ projectedLipLineMask: UVLipAlphaMask,
        minimumOverlapRatio: CGFloat = 0.22,
        fillScale: CGFloat = 0.74,
        supportRadius: Int = 5,
        expansionRadius: Int = 1,
        blurRadius: Int = 1
    ) -> (mask: UVLipAlphaMask, applied: Bool, overlapRatio: CGFloat, addedCoverage: CGFloat) {
        guard
            width == projectedLipLineMask.width,
            height == projectedLipLineMask.height,
            alpha.count == width * height,
            projectedLipLineMask.alpha.count == alpha.count
        else {
            return (self, false, 0, 0)
        }

        let baseThreshold: UInt8 = 32
        let lineThreshold: UInt8 = 24
        var baseActive = 0
        var lineActive = 0
        var overlap = 0
        for index in alpha.indices {
            let base = alpha[index] > baseThreshold
            let line = projectedLipLineMask.alpha[index] > lineThreshold
            if base { baseActive += 1 }
            if line { lineActive += 1 }
            if base && line { overlap += 1 }
        }

        guard baseActive > 32, lineActive > 32 else {
            return (self, false, 0, 0)
        }

        let overlapRatio = CGFloat(overlap) / CGFloat(max(min(baseActive, lineActive), 1))
        guard overlapRatio >= minimumOverlapRatio else {
            return (self, false, overlapRatio, 0)
        }

        let support = Self.softlyExpanded(
            alpha: alpha,
            width: width,
            height: height,
            radius: max(supportRadius, 0)
        )
        var lineFill = Self.softlyExpanded(
            alpha: projectedLipLineMask.alpha,
            width: width,
            height: height,
            radius: max(expansionRadius, 0)
        )
        lineFill = Self.blurred(alpha: lineFill, width: width, height: height, radius: max(blurRadius, 0))

        let clampedScale = min(max(fillScale, 0), 1)
        var output = alpha
        var addedPixels = 0
        for index in alpha.indices where lineFill[index] > lineThreshold {
            guard support[index] > 0 || alpha[index] > 0 else { continue }

            let candidate = UInt8(min(max(CGFloat(lineFill[index]) * clampedScale, 0), 236))
            if candidate > output[index] {
                if output[index] <= baseThreshold {
                    addedPixels += 1
                }
                output[index] = candidate
            }
        }

        let addedCoverage = CGFloat(addedPixels) / CGFloat(max(alpha.count, 1))
        guard addedPixels > 0 else {
            return (self, false, overlapRatio, 0)
        }

        return (UVLipAlphaMask(width: width, height: height, alpha: output), true, overlapRatio, addedCoverage)
    }

    func colorTrimmed(
        by colorSupportMask: UVLipAlphaMask,
        minimumOverlapRatio: CGFloat = 0.18,
        unsupportedRetain: CGFloat = 0.30,
        supportExpansionRadius: Int = 1,
        blurRadius: Int = 1
    ) -> (mask: UVLipAlphaMask, applied: Bool, overlapRatio: CGFloat, removedCoverage: CGFloat) {
        guard
            width == colorSupportMask.width,
            height == colorSupportMask.height,
            alpha.count == width * height,
            colorSupportMask.alpha.count == alpha.count
        else {
            return (self, false, 0, 0)
        }

        let baseThreshold: UInt8 = 32
        let supportThreshold: UInt8 = 44
        var baseActive = 0
        var overlap = 0
        for index in alpha.indices where alpha[index] > baseThreshold {
            baseActive += 1
            if colorSupportMask.alpha[index] > supportThreshold {
                overlap += 1
            }
        }

        guard baseActive > 32 else { return (self, false, 0, 0) }

        let overlapRatio = CGFloat(overlap) / CGFloat(baseActive)
        guard overlapRatio >= minimumOverlapRatio else {
            return (self, false, overlapRatio, 0)
        }

        var support = Self.softlyExpanded(
            alpha: colorSupportMask.alpha,
            width: width,
            height: height,
            radius: max(supportExpansionRadius, 0)
        )
        support = Self.blurred(alpha: support, width: width, height: height, radius: max(blurRadius, 0))

        let retain = min(max(unsupportedRetain, 0.30), 0.82)
        var output = alpha
        var removed = 0
        for index in alpha.indices {
            let baseAlpha = CGFloat(alpha[index])
            guard baseAlpha > 0 else { continue }

            let supportAmount = Self.smoothstep(CGFloat(support[index]) / 255)
            guard supportAmount < 0.12 else {
                output[index] = alpha[index]
                continue
            }

            let factor = retain + ((1 - retain) * supportAmount)
            let trimmed = UInt8(min(max(baseAlpha * factor, 0), 255))
            if alpha[index] > baseThreshold, trimmed <= baseThreshold {
                removed += 1
            }
            output[index] = min(alpha[index], trimmed)
        }

        let removedCoverage = CGFloat(removed) / CGFloat(max(alpha.count, 1))
        return (UVLipAlphaMask(width: width, height: height, alpha: output), true, overlapRatio, removedCoverage)
    }

    func colorEdgeSnapped(
        by colorSupportMask: UVLipAlphaMask,
        minimumOverlapRatio: CGFloat = 0.18,
        edgeAlphaThreshold: UInt8 = 150,
        unsupportedRetain: CGFloat = 0.18,
        supportExpansionRadius: Int = 2,
        blurRadius: Int = 1
    ) -> (mask: UVLipAlphaMask, applied: Bool, overlapRatio: CGFloat, changedCoverage: CGFloat) {
        guard
            width == colorSupportMask.width,
            height == colorSupportMask.height,
            alpha.count == width * height,
            colorSupportMask.alpha.count == alpha.count
        else {
            return (self, false, 0, 0)
        }

        let baseThreshold: UInt8 = 32
        let supportThreshold: UInt8 = 44
        var baseActive = 0
        var overlap = 0
        for index in alpha.indices where alpha[index] > baseThreshold {
            baseActive += 1
            if colorSupportMask.alpha[index] > supportThreshold {
                overlap += 1
            }
        }

        guard baseActive > 32 else { return (self, false, 0, 0) }

        let overlapRatio = CGFloat(overlap) / CGFloat(baseActive)
        guard overlapRatio >= minimumOverlapRatio else {
            return (self, false, overlapRatio, 0)
        }

        var support = Self.softlyExpanded(
            alpha: colorSupportMask.alpha,
            width: width,
            height: height,
            radius: max(supportExpansionRadius, 0)
        )
        support = Self.blurred(alpha: support, width: width, height: height, radius: max(blurRadius, 0))

        let retain = min(max(unsupportedRetain, 0.10), 0.55)
        let edgeLimit = max(edgeAlphaThreshold, baseThreshold)
        var output = alpha
        var changed = 0

        for index in alpha.indices {
            let baseAlpha = alpha[index]
            guard baseAlpha > baseThreshold, baseAlpha <= edgeLimit else { continue }

            let supportAmount = Self.smoothstep(CGFloat(support[index]) / 255)
            guard supportAmount < 0.98 else { continue }

            let factor = retain + ((1 - retain) * supportAmount)
            let snapped = UInt8(min(max(CGFloat(baseAlpha) * factor, 0), 255))
            if snapped < baseAlpha {
                output[index] = snapped
                changed += 1
            }
        }

        let changedCoverage = CGFloat(changed) / CGFloat(max(alpha.count, 1))
        guard changed > 0 else {
            return (self, false, overlapRatio, 0)
        }

        return (UVLipAlphaMask(width: width, height: height, alpha: output), true, overlapRatio, changedCoverage)
    }

    func smoothLipLineSilhouette(threshold: UInt8 = 32, radius: Int = 1) -> UVLipAlphaMask {
        guard width > 2, height > 2, alpha.count == width * height else { return self }

        let passRadius = max(radius, 1)
        let firstPass = Self.majorityFiltered(alpha: alpha, width: width, height: height, threshold: threshold, radius: passRadius)
        let secondPass = Self.majorityFiltered(alpha: firstPass, width: width, height: height, threshold: threshold, radius: passRadius)
        var output = [UInt8](repeating: 0, count: alpha.count)
        for index in alpha.indices where secondPass[index] > threshold {
            output[index] = max(alpha[index], 220)
        }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func softenedStroke(expansionRadius: Int = 1, blurRadius: Int = 1) -> UVLipAlphaMask {
        guard width > 2, height > 2, alpha.count == width * height else { return self }

        var output = Self.softlyExpanded(alpha: alpha, width: width, height: height, radius: max(expansionRadius, 0))
        output = Self.blurred(alpha: output, width: width, height: height, radius: max(blurRadius, 0))
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func outline(threshold: UInt8 = 32, widthPixels: Int = 1) -> UVLipAlphaMask {
        guard width > 2, height > 2, alpha.count == width * height else { return self }

        var output = [UInt8](repeating: 0, count: alpha.count)
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard alpha[index] > threshold else { continue }

                var touchesOutside = x == 0 || y == 0 || x == width - 1 || y == height - 1
                if !touchesOutside {
                    for sampleY in (y - 1)...(y + 1) {
                        for sampleX in (x - 1)...(x + 1) where sampleX != x || sampleY != y {
                            if alpha[sampleY * width + sampleX] <= threshold {
                                touchesOutside = true
                                break
                            }
                        }
                        if touchesOutside { break }
                    }
                }

                if touchesOutside {
                    output[index] = 255
                }
            }
        }

        let extraWidth = max(widthPixels - 1, 0)
        if extraWidth > 0 {
            output = Self.softlyExpanded(alpha: output, width: width, height: height, radius: extraWidth)
        }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func outerSilhouetteOutline(threshold: UInt8 = 32, widthPixels: Int = 1) -> UVLipAlphaMask {
        guard width > 2, height > 2, alpha.count == width * height else { return self }

        var topByColumn = [Int](repeating: Int.max, count: width)
        var bottomByColumn = [Int](repeating: -1, count: width)
        var activeColumns: [Int] = []

        for x in 0..<width {
            for y in 0..<height where alpha[y * width + x] > threshold {
                topByColumn[x] = min(topByColumn[x], y)
                bottomByColumn[x] = max(bottomByColumn[x], y)
            }
            if bottomByColumn[x] >= 0 {
                activeColumns.append(x)
            }
        }

        guard let minX = activeColumns.first, let maxX = activeColumns.last, maxX > minX else {
            return UVLipAlphaMask(width: width, height: height, alpha: [UInt8](repeating: 0, count: alpha.count))
        }

        let smoothedTop = Self.smoothedColumnBounds(topByColumn, activeColumns: activeColumns)
        let smoothedBottom = Self.smoothedColumnBounds(bottomByColumn, activeColumns: activeColumns)
        let radius = max(widthPixels - 1, 0)
        var output = [UInt8](repeating: 0, count: alpha.count)

        func setPixel(_ x: Int, _ y: Int) {
            guard x >= 0, x < width, y >= 0, y < height else { return }
            output[y * width + x] = 255
        }

        for x in minX...maxX where smoothedBottom[x] >= smoothedTop[x] {
            for offset in -radius...radius {
                setPixel(x, smoothedTop[x] + offset)
                setPixel(x, smoothedBottom[x] + offset)
            }
        }

        for edgeX in [minX, maxX] where smoothedBottom[edgeX] >= smoothedTop[edgeX] {
            for y in smoothedTop[edgeX]...smoothedBottom[edgeX] {
                for offset in -radius...radius {
                    setPixel(edgeX + offset, y)
                }
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func lipLineDebugOutline(threshold: UInt8 = 32, widthPixels: Int = 1) -> UVLipAlphaMask {
        lowerContourClampedForLipLine(
            threshold: threshold,
            centerInsetRatio: 0.10,
            sideInsetRatio: 0.02,
            fadePixels: 1
        )
        .smoothLipLineSilhouette(threshold: threshold, radius: 1)
        .curvedSilhouetteOutline(threshold: threshold, widthPixels: widthPixels)
    }

    func softenedForMakeup(
        edgeThreshold: UInt8 = 32,
        innerBlurRadius: Int = 1,
        outerFeatherRadius: Int = 4
    ) -> UVLipAlphaMask {
        guard width > 2, height > 2, alpha.count == width * height else { return self }

        let regularized = smoothLipLineSilhouette(threshold: edgeThreshold, radius: 1)
        let core = regularized.alpha.map { $0 > edgeThreshold ? $0 : 0 }
        let edgeBlurRadius = max(max(innerBlurRadius, outerFeatherRadius / 2), 0)
        let edge = Self.blurred(alpha: core, width: width, height: height, radius: edgeBlurRadius)
        let outsideCap: CGFloat = 72

        var output = [UInt8](repeating: 0, count: alpha.count)
        for index in alpha.indices {
            if core[index] > edgeThreshold {
                output[index] = max(alpha[index], edge[index])
            } else {
                output[index] = UInt8(min(CGFloat(edge[index]), outsideCap))
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func productContourSmoothedForMakeup(
        edgeThreshold: UInt8 = 32,
        smoothingIterations: Int = 2,
        edgeFeatherPixels: CGFloat = 1.0,
        outputScale: Int = 1
    ) -> UVLipAlphaMask {
        guard width > 4, height > 4, alpha.count == width * height else { return self }

        let components = Self.primaryLipComponents(
            alpha: alpha,
            width: width,
            height: height,
            threshold: edgeThreshold
        )
        guard !components.isEmpty else { return self }

        var filtered = [UInt8](repeating: 0, count: alpha.count)
        for component in components {
            for index in component.pixels {
                filtered[index] = max(filtered[index], alpha[index])
            }
        }

        guard filtered.contains(where: { $0 > edgeThreshold }) else { return self }

        let scale = max(outputScale, 1)
        let renderWidth = width * scale
        let renderHeight = height * scale
        let scaled = scale > 1
            ? Self.bilinearUpscaled(alpha: filtered, width: width, height: height, scale: scale)
            : filtered
        let gentlySmoothed = Self.blurred(
            alpha: scaled,
            width: renderWidth,
            height: renderHeight,
            radius: scale >= 3 ? 1 : 0
        )
        let edgeWidth = edgeFeatherPixels > 0.01 ? edgeFeatherPixels * CGFloat(scale) : 0.72
        let antialiased = Self.edgeAntialiased(
            alpha: gentlySmoothed,
            width: renderWidth,
            height: renderHeight,
            edgePixels: edgeWidth
        )

        return UVLipAlphaMask(width: renderWidth, height: renderHeight, alpha: antialiased)
    }

    func productRenderMaskForMakeup(
        targetSize: Int = 1024,
        preserveClosedMouthCrease: Bool = true
    ) -> UVLipAlphaMask {
        guard width > 4, height > 4, alpha.count == width * height else { return self }

        let signature = productRenderSignature(
            targetSize: targetSize,
            preserveClosedMouthCrease: preserveClosedMouthCrease
        )
        Self.productRenderCacheLock.lock()
        if let cached = Self.productRenderCache, cached.signature == signature {
            Self.productRenderCacheLock.unlock()
            return cached.mask
        }
        Self.productRenderCacheLock.unlock()

        let scale = max(1, min(4, (targetSize + max(width, height) - 1) / max(width, height)))
        let silhouette = preserveClosedMouthCrease ? preservedClosedMouthCrease() : self
        let smoothed = silhouette
            .productContourSmoothedForMakeup(
                edgeThreshold: 72,
                smoothingIterations: 1,
                edgeFeatherPixels: 0.55,
                outputScale: scale
            )
        let rendered = preserveClosedMouthCrease
            ? smoothed.preservedClosedMouthCrease(threshold: 36)
            : smoothed

        Self.productRenderCacheLock.lock()
        Self.productRenderCache = ProductRenderCacheEntry(signature: signature, mask: rendered)
        Self.productRenderCacheLock.unlock()

        return rendered
    }

    func productAlphaSmoothedForMakeup(edgeThreshold: UInt8 = 24) -> UVLipAlphaMask {
        guard width > 4, height > 4, alpha.count == width * height else { return self }

        let components = Self.primaryLipComponents(
            alpha: alpha,
            width: width,
            height: height,
            threshold: edgeThreshold
        )
        guard !components.isEmpty else { return self }

        var filtered = [UInt8](repeating: 0, count: alpha.count)
        for component in components {
            for index in component.pixels {
                filtered[index] = max(filtered[index], alpha[index])
            }
        }

        let firstBlur = Self.blurred(alpha: filtered, width: width, height: height, radius: 1)
        let secondBlur = Self.blurred(alpha: firstBlur, width: width, height: height, radius: 1)
        var output = filtered

        for index in filtered.indices {
            let source = filtered[index]
            let smooth = secondBlur[index]
            if source > edgeThreshold {
                output[index] = max(source, UInt8(min(Int(smooth), 245)))
            } else if smooth > 0 {
                output[index] = UInt8(min(Int(smooth), 96))
            }
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    private static func mergeMasks(_ masks: [UVLipAlphaMask]) -> UVLipAlphaMask? {
        guard let first = masks.first else { return nil }

        var merged = first.alpha
        for mask in masks.dropFirst() {
            guard mask.width == first.width, mask.height == first.height, mask.alpha.count == merged.count else {
                continue
            }
            for index in merged.indices {
                merged[index] = max(merged[index], mask.alpha[index])
            }
        }

        return UVLipAlphaMask(width: first.width, height: first.height, alpha: merged)
    }

    func subtracting(_ mask: UVLipAlphaMask) -> UVLipAlphaMask {
        guard
            width == mask.width,
            height == mask.height,
            alpha.count == mask.alpha.count
        else {
            return self
        }

        var output = alpha
        for index in output.indices {
            let remaining = Int(output[index]) - Int(mask.alpha[index])
            output[index] = UInt8(max(remaining, 0))
        }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func merged(with mask: UVLipAlphaMask) -> UVLipAlphaMask {
        guard
            width == mask.width,
            height == mask.height,
            alpha.count == mask.alpha.count
        else {
            return self
        }

        var output = alpha
        for index in output.indices {
            output[index] = max(output[index], mask.alpha[index])
        }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func clipped(to mask: UVLipAlphaMask) -> UVLipAlphaMask {
        guard
            width == mask.width,
            height == mask.height,
            alpha.count == mask.alpha.count
        else {
            return self
        }

        var output = alpha
        for index in output.indices {
            output[index] = min(output[index], mask.alpha[index])
        }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    func expanded(
        threshold: UInt8 = 24,
        radius: Int = 1,
        blurRadius: Int = 0
    ) -> UVLipAlphaMask {
        guard radius > 0 || blurRadius > 0 else { return self }

        var core = alpha.map { $0 > threshold ? $0 : 0 }
        if radius > 0 {
            core = Self.softlyExpanded(
                alpha: core,
                width: width,
                height: height,
                radius: radius
            )
        }
        if blurRadius > 0 {
            core = Self.blurred(
                alpha: core,
                width: width,
                height: height,
                radius: blurRadius
            )
        }
        return UVLipAlphaMask(width: width, height: height, alpha: core)
    }

    func enclosedHoleMask(
        threshold: UInt8 = 24,
        expansionRadius: Int = 1,
        blurRadius: Int = 1
    ) -> UVLipAlphaMask? {
        guard let holeMask = Self.enclosedBackgroundMask(
            alpha: alpha,
            width: width,
            height: height,
            threshold: threshold,
            outputScale: 1
        ) else {
            return nil
        }

        var output = holeMask.alpha
        if expansionRadius > 0 {
            output = Self.softlyExpanded(
                alpha: output,
                width: width,
                height: height,
                radius: expansionRadius
            )
        }
        if blurRadius > 0 {
            output = Self.blurred(
                alpha: output,
                width: width,
                height: height,
                radius: blurRadius
            )
        }
        return UVLipAlphaMask(width: width, height: height, alpha: output)
    }

    private static func enclosedBackgroundMask(
        alpha: [UInt8],
        width: Int,
        height: Int,
        threshold: UInt8,
        outputScale: Int
    ) -> UVLipAlphaMask? {
        guard width > 4, height > 4, alpha.count == width * height else { return nil }

        let foreground = alpha.map { $0 > threshold }
        var visited = [Bool](repeating: false, count: alpha.count)
        var queue: [Int] = []

        func enqueueIfBackground(_ x: Int, _ y: Int) {
            guard x >= 0, x < width, y >= 0, y < height else { return }
            let index = y * width + x
            guard !foreground[index], !visited[index] else { return }
            visited[index] = true
            queue.append(index)
        }

        for x in 0..<width {
            enqueueIfBackground(x, 0)
            enqueueIfBackground(x, height - 1)
        }
        for y in 0..<height {
            enqueueIfBackground(0, y)
            enqueueIfBackground(width - 1, y)
        }

        var head = 0
        while head < queue.count {
            let index = queue[head]
            head += 1
            let x = index % width
            let y = index / width
            enqueueIfBackground(x - 1, y)
            enqueueIfBackground(x + 1, y)
            enqueueIfBackground(x, y - 1)
            enqueueIfBackground(x, y + 1)
        }

        var holes = [UInt8](repeating: 0, count: alpha.count)
        var holeCount = 0
        for index in holes.indices where !foreground[index] && !visited[index] {
            holes[index] = 255
            holeCount += 1
        }

        guard holeCount >= max(6, (width * height) / 1600) else { return nil }

        let scale = max(outputScale, 1)
        let scaled = scale > 1
            ? bilinearUpscaled(alpha: holes, width: width, height: height, scale: scale)
            : holes
        let renderWidth = width * scale
        let renderHeight = height * scale
        let antialiased = edgeAntialiased(
            alpha: scaled,
            width: renderWidth,
            height: renderHeight,
            edgePixels: 0.65
        )
        return UVLipAlphaMask(width: renderWidth, height: renderHeight, alpha: antialiased)
    }

    private static func curvedFilledSilhouette(
        alpha: [UInt8],
        width: Int,
        height: Int,
        threshold: UInt8,
        smoothingIterations: Int,
        outputScale: Int,
        edgeFeatherPixels: CGFloat
    ) -> UVLipAlphaMask? {
        guard width > 4, height > 4, alpha.count == width * height else { return nil }

        var top = [Int](repeating: Int.max, count: width)
        var bottom = [Int](repeating: -1, count: width)
        var activeColumns: [Int] = []
        for x in 0..<width {
            for y in 0..<height where alpha[y * width + x] > threshold {
                top[x] = min(top[x], y)
                bottom[x] = max(bottom[x], y)
            }
            if bottom[x] >= top[x] {
                activeColumns.append(x)
            }
        }

        guard activeColumns.count >= 6 else { return nil }

        let smoothedTop = smoothedColumnBounds(top, activeColumns: activeColumns)
        let smoothedBottom = smoothedColumnBounds(bottom, activeColumns: activeColumns)
        let sampleStep = max(activeColumns.count / 48, 1)
        var sampledColumns = activeColumns.enumerated().compactMap { index, x in
            index.isMultiple(of: sampleStep) ? x : nil
        }
        if let last = activeColumns.last, sampledColumns.last != last {
            sampledColumns.append(last)
        }

        let scale = max(outputScale, 1)
        let renderWidth = width * scale
        let renderHeight = height * scale
        let scaleValue = CGFloat(scale)

        let topPoints = sampledColumns.map {
            CGPoint(
                x: (CGFloat($0) + 0.5) * scaleValue,
                y: (CGFloat(smoothedTop[$0]) + 0.5) * scaleValue
            )
        }
        let bottomPoints = sampledColumns.reversed().map {
            CGPoint(
                x: (CGFloat($0) + 0.5) * scaleValue,
                y: (CGFloat(smoothedBottom[$0]) + 0.5) * scaleValue
            )
        }

        let contour = chaikinSmoothed(
            points: topPoints + bottomPoints,
            iterations: max(smoothingIterations, 1)
        )
        guard contour.count >= 8 else { return nil }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        let renderSize = CGSize(width: renderWidth, height: renderHeight)
        let image = UIGraphicsImageRenderer(size: renderSize, format: format).image { context in
            UIColor.white.setFill()
            let path = UIBezierPath()
            path.move(to: contour[0])
            for point in contour.dropFirst() {
                path.addLine(to: point)
            }
            path.close()
            path.lineJoinStyle = .round
            path.lineCapStyle = .round
            path.fill()
        }

        guard let cgImage = image.cgImage else { return nil }

        var rgba = [UInt8](repeating: 0, count: renderWidth * renderHeight * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: renderWidth,
            height: renderHeight,
            bitsPerComponent: 8,
            bytesPerRow: renderWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(origin: .zero, size: renderSize))

        var output = [UInt8](repeating: 0, count: renderWidth * renderHeight)
        for index in output.indices {
            output[index] = rgba[index * 4 + 3]
        }

        if edgeFeatherPixels > 0.01 {
            let blurRadius = max(1, Int(round(edgeFeatherPixels * CGFloat(scale) * 0.4)))
            output = blurred(alpha: output, width: renderWidth, height: renderHeight, radius: blurRadius)
        } else if scale == 1 {
            output = edgeAntialiased(alpha: output, width: renderWidth, height: renderHeight, edgePixels: 0.5)
        }

        return UVLipAlphaMask(width: renderWidth, height: renderHeight, alpha: output)
    }

    func constrainedToLipRegion(
        _ lipRegionMask: UVLipAlphaMask,
        minimumOverlapRatio: CGFloat = 0.10,
        expansionRadius: Int = 1,
        blurRadius: Int = 0
    ) -> (mask: UVLipAlphaMask, applied: Bool, overlapRatio: CGFloat) {
        guard
            width == lipRegionMask.width,
            height == lipRegionMask.height,
            alpha.count == width * height,
            lipRegionMask.alpha.count == alpha.count
        else {
            return (self, false, 0)
        }

        let sourceThreshold: UInt8 = 24
        let regionThreshold: UInt8 = 20
        var activeCount = 0
        var overlapCount = 0
        for index in alpha.indices where alpha[index] > sourceThreshold {
            activeCount += 1
            if lipRegionMask.alpha[index] > regionThreshold {
                overlapCount += 1
            }
        }

        guard activeCount > 16 else { return (self, false, 0) }
        let overlapRatio = CGFloat(overlapCount) / CGFloat(activeCount)
        guard overlapRatio >= minimumOverlapRatio else {
            return (self, false, overlapRatio)
        }

        var gate = Self.softlyExpanded(
            alpha: lipRegionMask.alpha,
            width: width,
            height: height,
            radius: max(expansionRadius, 0)
        )
        gate = Self.blurred(alpha: gate, width: width, height: height, radius: max(blurRadius, 0))

        var output = [UInt8](repeating: 0, count: alpha.count)
        for index in alpha.indices {
            let gateAmount = Self.smoothstep(CGFloat(gate[index]) / 255)
            guard gateAmount > 0 else { continue }
            output[index] = UInt8(min(max(CGFloat(alpha[index]) * gateAmount, 0), 255))
        }

        return (UVLipAlphaMask(width: width, height: height, alpha: output), true, overlapRatio)
    }

    func lineFused(
        with colorSupportMask: UVLipAlphaMask,
        modelWeight: CGFloat = 0.50,
        colorWeight: CGFloat = 0.50,
        unsupportedModelRetain: CGFloat = 0.58,
        cornerProtectionRatio: CGFloat = 0.20,
        supportExpansionRadius: Int = 1,
        blurRadius: Int = 1
    ) -> UVLipAlphaMask {
        guard
            width == colorSupportMask.width,
            height == colorSupportMask.height,
            alpha.count == width * height,
            colorSupportMask.alpha.count == alpha.count
        else {
            return self
        }

        var support = Self.softlyExpanded(
            alpha: colorSupportMask.alpha,
            width: width,
            height: height,
            radius: max(supportExpansionRadius, 0)
        )
        support = Self.blurred(alpha: support, width: width, height: height, radius: max(blurRadius, 0))

        let clampedModelWeight = min(max(modelWeight, 0), 1)
        let clampedColorWeight = min(max(colorWeight, 0), 1)
        let clampedRetain = min(max(unsupportedModelRetain, 0.25), 1)
        let clampedCornerRatio = min(max(cornerProtectionRatio, 0), 0.45)
        let activeXRange = activeXRange(threshold: 32)
        let cornerProtectionWidth = activeXRange.map {
            max(CGFloat($0.upperBound - $0.lowerBound) * clampedCornerRatio, 1)
        } ?? 0
        var output = [UInt8](repeating: 0, count: alpha.count)

        for index in alpha.indices {
            let supportAmount = Self.smoothstep(CGFloat(support[index]) / 255)
            let modelOnly = CGFloat(alpha[index])
            let colorOnly = CGFloat(colorSupportMask.alpha[index])
            let combinedWeight = max(clampedModelWeight + clampedColorWeight, 0.001)
            let midpointVote = ((modelOnly * clampedModelWeight) + (colorOnly * clampedColorWeight)) / combinedWeight
            let modelRetained = modelOnly * clampedRetain
            let blended = modelRetained + ((midpointVote - modelRetained) * supportAmount)
            let cornerProtection: CGFloat
            if let activeXRange, cornerProtectionWidth > 0 {
                let x = CGFloat(index % width)
                let leftDistance = max(0, x - CGFloat(activeXRange.lowerBound))
                let rightDistance = max(0, CGFloat(activeXRange.upperBound) - x)
                let sideDistance = min(leftDistance, rightDistance)
                cornerProtection = 1 - min(max(sideDistance / cornerProtectionWidth, 0), 1)
            } else {
                cornerProtection = 0
            }
            let protected = blended + ((modelOnly - blended) * cornerProtection)
            output[index] = UInt8(min(max(protected, 0), 255))
        }

        return UVLipAlphaMask(
            width: width,
            height: height,
            alpha: Self.blurred(alpha: output, width: width, height: height, radius: 1)
        )
    }

    func cornerMirrorRepaired(
        against reference: UVLipAlphaMask,
        threshold: UInt8 = 32,
        cornerRegionRatio: CGFloat = 0.24,
        badCornerShiftRatio: CGFloat = 0.12,
        goodCornerShiftRatio: CGFloat = 0.08
    ) -> (mask: UVLipAlphaMask, applied: Bool, side: String) {
        guard
            width == reference.width,
            height == reference.height,
            alpha.count == width * height,
            reference.alpha.count == alpha.count,
            let referenceBounds = activeBounds(threshold: threshold),
            let candidateBounds = activeBounds(threshold: threshold)
        else {
            return (self, false, "")
        }

        let referenceWidth = max(referenceBounds.width, 1)
        let leftShift = abs(CGFloat(candidateBounds.minX - referenceBounds.minX)) / referenceWidth
        let rightShift = abs(CGFloat(candidateBounds.maxX - referenceBounds.maxX)) / referenceWidth
        let repairsLeft = leftShift >= badCornerShiftRatio && rightShift <= goodCornerShiftRatio
        let repairsRight = rightShift >= badCornerShiftRatio && leftShift <= goodCornerShiftRatio
        guard repairsLeft || repairsRight else {
            return (self, false, "")
        }

        let axisX = CGFloat(referenceBounds.minX + referenceBounds.maxX) * 0.5
        let cornerWidth = max(3, Int(round(referenceWidth * min(max(cornerRegionRatio, 0.08), 0.35))))
        let expandedReference = Self.softlyExpanded(
            alpha: reference.alpha,
            width: width,
            height: height,
            radius: 2
        )
        var output = alpha

        let targetXRange: ClosedRange<Int>
        let side: String
        if repairsLeft {
            let lower = max(0, min(candidateBounds.minX, referenceBounds.minX - 2))
            let upper = min(width - 1, referenceBounds.minX + cornerWidth)
            targetXRange = lower...max(lower, upper)
            side = "L"
        } else {
            let lower = max(0, referenceBounds.maxX - cornerWidth)
            let upper = min(width - 1, max(candidateBounds.maxX, referenceBounds.maxX + 2))
            targetXRange = min(lower, upper)...upper
            side = "R"
        }

        for y in 0..<height {
            for x in targetXRange {
                let index = y * width + x
                let mirrorX = Int(round((axisX * 2) - CGFloat(x)))
                guard mirrorX >= 0, mirrorX < width else {
                    output[index] = 0
                    continue
                }

                let mirroredAlpha = alpha[y * width + mirrorX]
                let gate = CGFloat(expandedReference[index]) / 255
                guard gate > 0 else {
                    output[index] = 0
                    continue
                }

                output[index] = UInt8(min(max(CGFloat(mirroredAlpha) * gate, 0), 255))
            }
        }

        return (UVLipAlphaMask(width: width, height: height, alpha: output), true, side)
    }

    private func activeXRange(threshold: UInt8) -> ClosedRange<Int>? {
        guard width > 0, height > 0, alpha.count == width * height else { return nil }

        var minX = width
        var maxX = -1
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] > threshold {
                minX = min(minX, x)
                maxX = max(maxX, x)
            }
        }

        guard maxX >= minX else { return nil }
        return minX...maxX
    }

    private struct ActiveBounds {
        let minX: Int
        let maxX: Int
        let minY: Int
        let maxY: Int

        var width: CGFloat {
            CGFloat(maxX - minX + 1)
        }
    }

    private func activeBounds(threshold: UInt8) -> ActiveBounds? {
        guard width > 0, height > 0, alpha.count == width * height else { return nil }

        var minX = width
        var maxX = -1
        var minY = height
        var maxY = -1
        var activeCount = 0
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] > threshold {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
                activeCount += 1
            }
        }

        guard activeCount >= 24, maxX >= minX, maxY >= minY else { return nil }
        return ActiveBounds(minX: minX, maxX: maxX, minY: minY, maxY: maxY)
    }

    private struct LipComponent {
        let pixels: [Int]
        let minX: Int
        let maxX: Int
        let minY: Int
        let maxY: Int

        var count: Int {
            pixels.count
        }

        var width: Int {
            maxX - minX + 1
        }

        var height: Int {
            maxY - minY + 1
        }

        func horizontalOverlapRatio(with other: LipComponent) -> CGFloat {
            let overlap = max(0, min(maxX, other.maxX) - max(minX, other.minX) + 1)
            return CGFloat(overlap) / CGFloat(max(min(width, other.width), 1))
        }

        func verticalGap(to other: LipComponent) -> Int {
            if maxY < other.minY {
                return other.minY - maxY
            }
            if other.maxY < minY {
                return minY - other.maxY
            }
            return 0
        }
    }

    private static func primaryLipComponents(
        alpha: [UInt8],
        width: Int,
        height: Int,
        threshold: UInt8
    ) -> [LipComponent] {
        guard width > 0, height > 0, alpha.count == width * height else { return [] }

        var visited = [Bool](repeating: false, count: alpha.count)
        var components: [LipComponent] = []
        let neighborOffsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1)
        ]

        for start in alpha.indices where alpha[start] > threshold && !visited[start] {
            var queue = [start]
            var head = 0
            var pixels: [Int] = []
            var minX = width
            var maxX = -1
            var minY = height
            var maxY = -1
            visited[start] = true

            while head < queue.count {
                let index = queue[head]
                head += 1
                pixels.append(index)

                let x = index % width
                let y = index / width
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)

                for offset in neighborOffsets {
                    let sampleX = x + offset.0
                    let sampleY = y + offset.1
                    guard sampleX >= 0, sampleX < width, sampleY >= 0, sampleY < height else {
                        continue
                    }
                    let sampleIndex = sampleY * width + sampleX
                    guard !visited[sampleIndex], alpha[sampleIndex] > threshold else { continue }

                    visited[sampleIndex] = true
                    queue.append(sampleIndex)
                }
            }

            guard pixels.count >= 4 else { continue }
            components.append(LipComponent(pixels: pixels, minX: minX, maxX: maxX, minY: minY, maxY: maxY))
        }

        let sorted = components.sorted { $0.count > $1.count }
        guard let main = sorted.first else { return [] }

        let minimumSecondaryCount = max(24, main.count / 3)
        let maximumVerticalGap = max(main.height, height / 8)
        return sorted.filter { component in
            if component.count == main.count,
               component.minX == main.minX,
               component.maxX == main.maxX,
               component.minY == main.minY,
               component.maxY == main.maxY {
                return true
            }

            return component.count >= minimumSecondaryCount
                && component.horizontalOverlapRatio(with: main) >= 0.35
                && component.verticalGap(to: main) <= maximumVerticalGap
        }
    }

    private static func edgeAntialiased(
        alpha: [UInt8],
        width: Int,
        height: Int,
        edgePixels: CGFloat
    ) -> [UInt8] {
        signedDistanceFeathered(
            alpha: alpha,
            width: width,
            height: height,
            featherPixels: edgePixels
        )
    }

    private static func bilinearUpscaled(
        alpha: [UInt8],
        width: Int,
        height: Int,
        scale: Int
    ) -> [UInt8] {
        guard scale > 1, width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        let scaledWidth = width * scale
        let scaledHeight = height * scale
        let invScale = CGFloat(1) / CGFloat(scale)
        var output = [UInt8](repeating: 0, count: scaledWidth * scaledHeight)

        for y in 0..<scaledHeight {
            let sourceY = (CGFloat(y) + 0.5) * invScale - 0.5
            let y0 = min(max(Int(floor(sourceY)), 0), height - 1)
            let y1 = min(y0 + 1, height - 1)
            let ty = min(max(sourceY - CGFloat(y0), 0), 1)

            for x in 0..<scaledWidth {
                let sourceX = (CGFloat(x) + 0.5) * invScale - 0.5
                let x0 = min(max(Int(floor(sourceX)), 0), width - 1)
                let x1 = min(x0 + 1, width - 1)
                let tx = min(max(sourceX - CGFloat(x0), 0), 1)

                let topLeft = CGFloat(alpha[y0 * width + x0])
                let topRight = CGFloat(alpha[y0 * width + x1])
                let bottomLeft = CGFloat(alpha[y1 * width + x0])
                let bottomRight = CGFloat(alpha[y1 * width + x1])
                let top = topLeft + ((topRight - topLeft) * tx)
                let bottom = bottomLeft + ((bottomRight - bottomLeft) * tx)
                let value = top + ((bottom - top) * ty)
                output[y * scaledWidth + x] = UInt8(min(max(value.rounded(), 0), 255))
            }
        }

        return output
    }

    private static func signedDistanceFeathered(
        alpha: [UInt8],
        width: Int,
        height: Int,
        featherPixels: CGFloat
    ) -> [UInt8] {
        guard featherPixels > 0, width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        let binary = alpha.map { $0 > 16 }
        let radius = max(2, Int(ceil(featherPixels * 2)) + 1)
        var output = [UInt8](repeating: 0, count: alpha.count)

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let inside = binary[index]
                var nearestOpposite = CGFloat.greatestFiniteMagnitude

                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                        let sampleIndex = sampleY * width + sampleX
                        guard binary[sampleIndex] != inside else { continue }
                        let distance = hypot(CGFloat(sampleX - x), CGFloat(sampleY - y))
                        nearestOpposite = min(nearestOpposite, distance)
                    }
                }

                if nearestOpposite == CGFloat.greatestFiniteMagnitude {
                    output[index] = inside ? 255 : 0
                    continue
                }

                let boundaryDistance = max(nearestOpposite - 0.5, 0)
                let signedDistance = inside ? boundaryDistance : -boundaryDistance
                let t = (signedDistance + featherPixels) / max(featherPixels * 2, 0.001)
                output[index] = UInt8(min(max(smoothstep(t) * 255, 0), 255))
            }
        }

        return output
    }

    private func curvedSilhouetteOutline(threshold: UInt8, widthPixels: Int) -> UVLipAlphaMask {
        guard width > 4, height > 4, alpha.count == width * height else {
            return outerSilhouetteOutline(threshold: threshold, widthPixels: widthPixels)
        }

        var top = [Int](repeating: Int.max, count: width)
        var bottom = [Int](repeating: -1, count: width)
        var activeColumns: [Int] = []
        for x in 0..<width {
            for y in 0..<height where alpha[y * width + x] > threshold {
                top[x] = min(top[x], y)
                bottom[x] = max(bottom[x], y)
            }
            if bottom[x] >= top[x] {
                activeColumns.append(x)
            }
        }

        guard activeColumns.count >= 6 else {
            return outerSilhouetteOutline(threshold: threshold, widthPixels: widthPixels)
        }

        let smoothedTop = Self.smoothedColumnBounds(top, activeColumns: activeColumns)
        let smoothedBottom = Self.smoothedColumnBounds(bottom, activeColumns: activeColumns)
        let sampleStep = max(activeColumns.count / 42, 1)
        var sampledColumns = activeColumns.enumerated().compactMap { index, x in
            index.isMultiple(of: sampleStep) ? x : nil
        }
        if let last = activeColumns.last, sampledColumns.last != last {
            sampledColumns.append(last)
        }

        let topPoints = sampledColumns.map {
            CGPoint(x: CGFloat($0) + 0.5, y: CGFloat(smoothedTop[$0]) + 0.5)
        }
        let bottomPoints = sampledColumns.reversed().map {
            CGPoint(x: CGFloat($0) + 0.5, y: CGFloat(smoothedBottom[$0]) + 0.5)
        }
        let contour = Self.chaikinSmoothed(points: topPoints + bottomPoints, iterations: 1)
        guard contour.count >= 6 else {
            return outerSilhouetteOutline(threshold: threshold, widthPixels: widthPixels)
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        let size = CGSize(width: width, height: height)
        let image = UIGraphicsImageRenderer(size: size, format: format).image { context in
            UIColor.white.setStroke()
            let path = UIBezierPath()
            path.move(to: contour[0])
            for point in contour.dropFirst() {
                path.addLine(to: point)
            }
            path.close()
            path.lineWidth = CGFloat(max(widthPixels, 1)) * 1.7
            path.lineJoinStyle = .round
            path.lineCapStyle = .round
            path.stroke()
        }

        guard let cgImage = image.cgImage else {
            return outerSilhouetteOutline(threshold: threshold, widthPixels: widthPixels)
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return outerSilhouetteOutline(threshold: threshold, widthPixels: widthPixels)
        }

        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        var output = [UInt8](repeating: 0, count: width * height)
        for index in output.indices {
            output[index] = rgba[index * 4 + 3]
        }

        return UVLipAlphaMask(width: width, height: height, alpha: output)
            .softenedStroke(expansionRadius: 1, blurRadius: 1)
    }

    private static func chaikinSmoothed(points: [CGPoint], iterations: Int) -> [CGPoint] {
        guard points.count >= 3, iterations > 0 else { return points }

        var current = points
        for _ in 0..<iterations {
            var next: [CGPoint] = []
            for index in current.indices {
                let p0 = current[index]
                let p1 = current[(index + 1) % current.count]
                next.append(CGPoint(
                    x: p0.x * 0.75 + p1.x * 0.25,
                    y: p0.y * 0.75 + p1.y * 0.25
                ))
                next.append(CGPoint(
                    x: p0.x * 0.25 + p1.x * 0.75,
                    y: p0.y * 0.25 + p1.y * 0.75
                ))
            }
            current = next
        }

        return current
    }

    private static func smoothedColumnBounds(_ bounds: [Int], activeColumns: [Int]) -> [Int] {
        guard activeColumns.count >= 5 else { return bounds }

        var output = bounds
        let smoothingRadius = activeColumns.count >= 18 ? 4 : 2
        for x in activeColumns {
            let samples = activeColumns
                .filter { abs($0 - x) <= smoothingRadius }
                .map { bounds[$0] }
                .sorted()
            guard !samples.isEmpty else { continue }
            output[x] = samples[samples.count / 2]
        }
        return output
    }

    private static func softlyExpanded(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else { return alpha }

        var output = alpha
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] == 0 {
                var neighborMax = 0
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                        neighborMax = max(neighborMax, Int(alpha[sampleY * width + sampleX]))
                    }
                }

                if neighborMax > 36 {
                    output[y * width + x] = UInt8(min(max(CGFloat(neighborMax) * 0.54, 0), 170))
                }
            }
        }

        return output
    }

    private static func blurred(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else { return alpha }

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

    private static func majorityFiltered(
        alpha: [UInt8],
        width: Int,
        height: Int,
        threshold: UInt8,
        radius: Int
    ) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else { return alpha }

        var output = [UInt8](repeating: 0, count: alpha.count)
        for y in 0..<height {
            for x in 0..<width {
                var active = 0
                var total = 0
                var maxAlpha = 0
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                        let value = Int(alpha[sampleY * width + sampleX])
                        if value > Int(threshold) {
                            active += 1
                            maxAlpha = max(maxAlpha, value)
                        }
                        total += 1
                    }
                }

                let required = max(3, Int(ceil(CGFloat(total) * 0.42)))
                if active >= required {
                    output[y * width + x] = UInt8(min(max(maxAlpha, 220), 255))
                }
            }
        }
        return output
    }

    private static func median(_ values: [Int]) -> Int {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private static func smoothstep(_ value: CGFloat) -> CGFloat {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }
}

struct UVLipMaskLockResult {
    let mask: UVLipAlphaMask
    let decision: UVLipMaskLockDecision
}

struct UVLipLineCandidateSelection {
    let mask: UVLipAlphaMask
    let status: String
}

struct UVLipCoverageGateResult {
    let passed: Bool
    let coverage: CGFloat
    let spillRate: CGFloat
    let areaRatio: CGFloat
    let iou: CGFloat

    var statusText: String {
        String(format: " cov%.2f spill%.2f area%.2f iou%.2f", coverage, spillRate, areaRatio, iou)
    }
}

enum UVLipCoverageGate {
    static func evaluate(
        reference: UVLipAlphaMask,
        candidate: UVLipAlphaMask,
        threshold: UInt8 = 32,
        minimumCoverage: CGFloat = 0.72,
        maximumSpillRate: CGFloat = 0.22,
        minimumAreaRatio: CGFloat = 0.70,
        maximumAreaRatio: CGFloat = 1.28,
        minimumIOU: CGFloat = 0.55
    ) -> UVLipCoverageGateResult {
        guard
            reference.width == candidate.width,
            reference.height == candidate.height,
            reference.alpha.count == candidate.alpha.count,
            !reference.alpha.isEmpty
        else {
            return UVLipCoverageGateResult(passed: false, coverage: 0, spillRate: 1, areaRatio: 0, iou: 0)
        }

        var referenceCount = 0
        var candidateCount = 0
        var intersectionCount = 0
        var unionCount = 0
        for index in reference.alpha.indices {
            let referenceActive = reference.alpha[index] > threshold
            let candidateActive = candidate.alpha[index] > threshold
            if referenceActive { referenceCount += 1 }
            if candidateActive { candidateCount += 1 }
            if referenceActive && candidateActive { intersectionCount += 1 }
            if referenceActive || candidateActive { unionCount += 1 }
        }

        guard referenceCount > 0, candidateCount > 0, unionCount > 0 else {
            return UVLipCoverageGateResult(passed: false, coverage: 0, spillRate: 1, areaRatio: 0, iou: 0)
        }

        let coverage = CGFloat(intersectionCount) / CGFloat(referenceCount)
        let spillRate = CGFloat(max(candidateCount - intersectionCount, 0)) / CGFloat(candidateCount)
        let areaRatio = CGFloat(candidateCount) / CGFloat(referenceCount)
        let iou = CGFloat(intersectionCount) / CGFloat(unionCount)
        let passed = coverage >= minimumCoverage
            && spillRate <= maximumSpillRate
            && areaRatio >= minimumAreaRatio
            && areaRatio <= maximumAreaRatio
            && iou >= minimumIOU
        return UVLipCoverageGateResult(
            passed: passed,
            coverage: coverage,
            spillRate: spillRate,
            areaRatio: areaRatio,
            iou: iou
        )
    }
}

struct UVLipLineGeometryGateResult {
    let passed: Bool
    let cornerShiftRatio: CGFloat
    let centerShiftRatio: CGFloat
    let sideBalanceRatio: CGFloat
    let widthRatio: CGFloat
    let heightRatio: CGFloat

    var statusText: String {
        String(
            format: " anchor c%.2f x%.2f s%.2f w%.2f h%.2f",
            cornerShiftRatio,
            centerShiftRatio,
            sideBalanceRatio,
            widthRatio,
            heightRatio
        )
    }
}

enum UVLipLineGeometryGate {
    static func evaluate(
        reference: UVLipAlphaMask,
        candidate: UVLipAlphaMask,
        threshold: UInt8 = 32,
        maximumCornerShiftRatio: CGFloat = 0.22,
        maximumCenterShiftRatio: CGFloat = 0.14,
        minimumSideBalanceRatio: CGFloat = 0.24,
        minimumWidthRatio: CGFloat = 0.55,
        maximumWidthRatio: CGFloat = 1.25,
        minimumHeightRatio: CGFloat = 0.42,
        maximumHeightRatio: CGFloat = 1.48
    ) -> UVLipLineGeometryGateResult {
        guard
            reference.width == candidate.width,
            reference.height == candidate.height,
            reference.alpha.count == candidate.alpha.count,
            let referenceShape = Shape(mask: reference, threshold: threshold, splitAxisX: nil),
            let candidateShape = Shape(mask: candidate, threshold: threshold, splitAxisX: referenceShape.centerX)
        else {
            return UVLipLineGeometryGateResult(
                passed: false,
                cornerShiftRatio: 1,
                centerShiftRatio: 1,
                sideBalanceRatio: 0,
                widthRatio: 0,
                heightRatio: 0
            )
        }

        let referenceWidth = max(referenceShape.width, 1)
        let referenceHeight = max(referenceShape.height, 1)
        let cornerShift = max(
            abs(CGFloat(candidateShape.minX - referenceShape.minX)),
            abs(CGFloat(candidateShape.maxX - referenceShape.maxX))
        ) / referenceWidth
        let centerShift = abs(candidateShape.centerX - referenceShape.centerX) / referenceWidth
        let widthRatio = candidateShape.width / referenceWidth
        let heightRatio = candidateShape.height / referenceHeight
        let sideBalance = candidateShape.sideBalanceRatio
        let passed = cornerShift <= maximumCornerShiftRatio
            && centerShift <= maximumCenterShiftRatio
            && sideBalance >= minimumSideBalanceRatio
            && widthRatio >= minimumWidthRatio
            && widthRatio <= maximumWidthRatio
            && heightRatio >= minimumHeightRatio
            && heightRatio <= maximumHeightRatio

        return UVLipLineGeometryGateResult(
            passed: passed,
            cornerShiftRatio: cornerShift,
            centerShiftRatio: centerShift,
            sideBalanceRatio: sideBalance,
            widthRatio: widthRatio,
            heightRatio: heightRatio
        )
    }

    private struct Shape {
        let minX: Int
        let maxX: Int
        let minY: Int
        let maxY: Int
        let centerX: CGFloat
        let sideBalanceRatio: CGFloat

        var width: CGFloat {
            CGFloat(maxX - minX + 1)
        }

        var height: CGFloat {
            CGFloat(maxY - minY + 1)
        }

        init?(mask: UVLipAlphaMask, threshold: UInt8, splitAxisX: CGFloat?) {
            guard mask.width > 0, mask.height > 0, mask.alpha.count == mask.width * mask.height else {
                return nil
            }

            var minX = mask.width
            var maxX = -1
            var minY = mask.height
            var maxY = -1
            var activeCount = 0
            var xSum: CGFloat = 0
            var leftCount = 0
            var rightCount = 0
            let axis = splitAxisX ?? CGFloat(mask.width) * 0.5

            for y in 0..<mask.height {
                for x in 0..<mask.width where mask.alpha[y * mask.width + x] > threshold {
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                    activeCount += 1
                    xSum += CGFloat(x)
                    if CGFloat(x) < axis {
                        leftCount += 1
                    } else {
                        rightCount += 1
                    }
                }
            }

            guard activeCount >= 24, maxX >= minX, maxY >= minY else { return nil }

            self.minX = minX
            self.maxX = maxX
            self.minY = minY
            self.maxY = maxY
            centerX = xSum / CGFloat(activeCount)
            let smallerSide = min(leftCount, rightCount)
            let largerSide = max(leftCount, rightCount)
            sideBalanceRatio = largerSide > 0 ? CGFloat(smallerSide) / CGFloat(largerSide) : 0
        }
    }
}

struct UVLipLineCandidateSelector {
    static func select(
        displayedMask: UVLipAlphaMask,
        colorSupportMask: UVLipAlphaMask?,
        allowColorSupport: Bool
    ) -> UVLipLineCandidateSelection {
        select(
            displayedMask: displayedMask,
            colorSupportMask: colorSupportMask,
            source: allowColorSupport ? .fused : .uvOnly
        )
    }

    static func select(
        displayedMask: UVLipAlphaMask,
        colorSupportMask: UVLipAlphaMask?,
        source: LipLineDebugSource
    ) -> UVLipLineCandidateSelection {
        switch source {
        case .none, .uvOnly:
            return UVLipLineCandidateSelection(mask: displayedMask, status: " lineFuse uvOnly")
        case .colorOnly:
            guard let colorSupportMask else {
                return UVLipLineCandidateSelection(mask: emptyLike(displayedMask), status: " lineColor nil")
            }

            let constrained = colorSupportMask.constrainedToLipRegion(
                displayedMask,
                minimumOverlapRatio: 0.18,
                expansionRadius: 2,
                blurRadius: 1
            )
            guard constrained.applied, isCredibleColorCandidate(constrained.mask, comparedWith: displayedMask) else {
                return UVLipLineCandidateSelection(
                    mask: emptyLike(displayedMask),
                    status: String(format: " lineColor reject%.2f", constrained.overlapRatio)
                )
            }

            return UVLipLineCandidateSelection(
                mask: constrained.mask,
                status: String(format: " lineColor only%.2f", constrained.overlapRatio)
            )
        case .fused:
            guard let colorSupportMask else {
                return UVLipLineCandidateSelection(mask: displayedMask, status: " lineFuse uvOnly")
            }

            let constrained = colorSupportMask.constrainedToLipRegion(
                displayedMask,
                minimumOverlapRatio: 0.24,
                expansionRadius: 2,
                blurRadius: 1
            )
            if constrained.applied, isCredibleColorCandidate(constrained.mask, comparedWith: displayedMask) {
                let repaired = constrained.mask.cornerMirrorRepaired(against: displayedMask)
                let candidateMask = repaired.mask
                let anchorGate = UVLipLineGeometryGate.evaluate(reference: displayedMask, candidate: candidateMask)
                guard anchorGate.passed else {
                    return UVLipLineCandidateSelection(
                        mask: displayedMask,
                        status: " lineFuse anchorOff" + anchorGate.statusText
                    )
                }
                let fusedMask = displayedMask.lineFused(with: candidateMask)
                let coverageGate = UVLipCoverageGate.evaluate(reference: displayedMask, candidate: fusedMask)
                guard coverageGate.passed else {
                    return UVLipLineCandidateSelection(
                        mask: displayedMask,
                        status: " lineFuse gateOff" + coverageGate.statusText
                    )
                }
                let repairStatus = repaired.applied ? " mirror\(repaired.side)" : ""
                return UVLipLineCandidateSelection(
                    mask: fusedMask,
                    status: String(format: " lineFuse mid%.2f", constrained.overlapRatio) + repairStatus + coverageGate.statusText
                )
            }

            return UVLipLineCandidateSelection(
                mask: displayedMask,
                status: String(format: " lineFuse uvOnly%.2f", constrained.overlapRatio)
            )
        }
    }

    private static func emptyLike(_ mask: UVLipAlphaMask) -> UVLipAlphaMask {
        UVLipAlphaMask(width: mask.width, height: mask.height, alpha: [UInt8](repeating: 0, count: mask.alpha.count))
    }

    private static func isCredibleColorCandidate(_ colorMask: UVLipAlphaMask, comparedWith displayedMask: UVLipAlphaMask) -> Bool {
        let displayedCoverage = displayedMask.nonzeroCoverage
        let colorCoverage = colorMask.nonzeroCoverage
        guard displayedCoverage > 0 else { return false }

        let coverageRatio = colorCoverage / displayedCoverage
        return coverageRatio >= 0.42 && coverageRatio <= 1.22
    }
}

enum UVLipMaskLockDecision: Equatable {
    case warmingUp(Int)
    case locked
    case held
    case pendingRelock(Int)
    case relocked
    case rejected

    var statusText: String {
        switch self {
        case .warmingUp(let count):
            "warm\(count)"
        case .locked:
            "lock"
        case .held:
            "hold"
        case .pendingRelock(let count):
            "cand\(count)"
        case .relocked:
            "relock"
        case .rejected:
            "reject"
        }
    }
}

struct UVLipMaskLock {
    var stableFramesRequired = 2
    var relockFramesRequired = 4
    var minimumCoverage: CGFloat = 0.002
    var maximumCoverage: CGFloat = 0.22
    var minimumAlpha: UInt8 = 64
    var stableIOU: CGFloat = 0.60
    var holdIOU: CGFloat = 0.65
    var relockIOU: CGFloat = 0.30
    var stableCoverageRatio: CGFloat = 0.70
    var holdCoverageRatio: CGFloat = 0.76
    var relockCoverageRatio: CGFloat = 0.45
    var stableCenterShiftPixels: CGFloat = 10
    var holdCenterShiftPixels: CGFloat = 10
    var relockCenterShiftPixels: CGFloat = 24
    var holdUpdateResponse: CGFloat = 0

    private var lockedMask: UVLipAlphaMask?
    private var pendingMask: UVLipAlphaMask?
    private var pendingCount = 0

    init(
        stableFramesRequired: Int = 2,
        relockFramesRequired: Int = 4
    ) {
        self.stableFramesRequired = max(stableFramesRequired, 1)
        self.relockFramesRequired = max(relockFramesRequired, 1)
    }

    mutating func reset() {
        lockedMask = nil
        pendingMask = nil
        pendingCount = 0
    }

    mutating func resolve(candidate: UVLipAlphaMask) -> UVLipMaskLockResult {
        guard isValid(candidate) else {
            pendingMask = nil
            pendingCount = 0
            return UVLipMaskLockResult(mask: lockedMask ?? candidate, decision: .rejected)
        }

        guard let lockedMask else {
            return warmUp(candidate: candidate)
        }

        if isHoldSimilar(lockedMask, candidate) {
            pendingMask = nil
            pendingCount = 0
            let response = min(max(holdUpdateResponse, 0), 1)
            let updatedMask = response > 0
                ? candidate.temporallySmoothed(previous: lockedMask, response: response)
                : lockedMask
            self.lockedMask = updatedMask
            return UVLipMaskLockResult(mask: updatedMask, decision: .held)
        }

        guard isPlausibleRelock(lockedMask, candidate) else {
            pendingMask = nil
            pendingCount = 0
            return UVLipMaskLockResult(mask: lockedMask, decision: .rejected)
        }

        if let pendingMask, isStable(pendingMask, candidate) {
            pendingCount += 1
        } else {
            pendingCount = 1
        }
        pendingMask = candidate

        if pendingCount >= relockFramesRequired {
            self.lockedMask = candidate
            pendingMask = nil
            pendingCount = 0
            return UVLipMaskLockResult(mask: candidate, decision: .relocked)
        }

        return UVLipMaskLockResult(mask: lockedMask, decision: .pendingRelock(pendingCount))
    }

    private mutating func warmUp(candidate: UVLipAlphaMask) -> UVLipMaskLockResult {
        if let pendingMask, isStable(pendingMask, candidate) {
            pendingCount += 1
        } else {
            pendingCount = 1
        }
        pendingMask = candidate

        if pendingCount >= stableFramesRequired {
            lockedMask = candidate
            pendingMask = nil
            pendingCount = 0
            return UVLipMaskLockResult(mask: candidate, decision: .locked)
        }

        return UVLipMaskLockResult(mask: candidate, decision: .warmingUp(pendingCount))
    }

    private func isValid(_ mask: UVLipAlphaMask) -> Bool {
        guard mask.maximumAlpha >= minimumAlpha, let metrics = metrics(mask) else { return false }
        return metrics.coverage >= minimumCoverage && metrics.coverage <= maximumCoverage
    }

    private func isStable(_ previous: UVLipAlphaMask, _ candidate: UVLipAlphaMask) -> Bool {
        guard let comparison = compare(previous, candidate) else { return false }
        return comparison.iou >= stableIOU
            && comparison.coverageRatio >= stableCoverageRatio
            && comparison.centerDistance <= stableCenterShiftPixels
    }

    private func isHoldSimilar(_ locked: UVLipAlphaMask, _ candidate: UVLipAlphaMask) -> Bool {
        guard let comparison = compare(locked, candidate) else { return false }
        return comparison.iou >= holdIOU
            && comparison.coverageRatio >= holdCoverageRatio
            && comparison.centerDistance <= holdCenterShiftPixels
    }

    private func isPlausibleRelock(_ locked: UVLipAlphaMask, _ candidate: UVLipAlphaMask) -> Bool {
        guard let comparison = compare(locked, candidate) else { return false }
        return comparison.iou >= relockIOU
            && comparison.coverageRatio >= relockCoverageRatio
            && comparison.centerDistance <= relockCenterShiftPixels
    }

    private func metrics(_ mask: UVLipAlphaMask, threshold: UInt8 = 32) -> MaskMetrics? {
        guard mask.width > 0, mask.height > 0, mask.alpha.count == mask.width * mask.height else {
            return nil
        }

        var active = 0
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask.alpha[y * mask.width + x] > threshold {
                active += 1
                sumX += CGFloat(x)
                sumY += CGFloat(y)
            }
        }

        guard active > 8 else { return nil }
        let count = CGFloat(active)
        return MaskMetrics(
            coverage: count / CGFloat(mask.alpha.count),
            center: CGPoint(x: sumX / count, y: sumY / count)
        )
    }

    private func compare(_ lhs: UVLipAlphaMask, _ rhs: UVLipAlphaMask, threshold: UInt8 = 32) -> MaskComparison? {
        guard
            lhs.width == rhs.width,
            lhs.height == rhs.height,
            lhs.alpha.count == rhs.alpha.count,
            let lhsMetrics = metrics(lhs, threshold: threshold),
            let rhsMetrics = metrics(rhs, threshold: threshold)
        else {
            return nil
        }

        var intersection = 0
        var union = 0
        for index in lhs.alpha.indices {
            let lhsActive = lhs.alpha[index] > threshold
            let rhsActive = rhs.alpha[index] > threshold
            if lhsActive && rhsActive {
                intersection += 1
            }
            if lhsActive || rhsActive {
                union += 1
            }
        }

        guard union > 0 else { return nil }
        let coverageRatio = min(lhsMetrics.coverage, rhsMetrics.coverage) / max(lhsMetrics.coverage, rhsMetrics.coverage)
        return MaskComparison(
            iou: CGFloat(intersection) / CGFloat(union),
            coverageRatio: coverageRatio,
            centerDistance: hypot(lhsMetrics.center.x - rhsMetrics.center.x, lhsMetrics.center.y - rhsMetrics.center.y)
        )
    }

    private struct MaskMetrics {
        let coverage: CGFloat
        let center: CGPoint
    }

    private struct MaskComparison {
        let iou: CGFloat
        let coverageRatio: CGFloat
        let centerDistance: CGFloat
    }
}

struct UVLipEdgeRefinement: Equatable {
    let radius: UInt32
    let insetPixels: Float
    let featherPixels: Float
    let hardThreshold: UInt8
    let maximumAlpha: Float
    let insideRetain: Float
    let outsideFeatherScale: Float

    static let product = UVLipEdgeRefinement(
        radius: 2,
        insetPixels: 0.45,
        featherPixels: 0.50,
        hardThreshold: 104,
        maximumAlpha: 255,
        insideRetain: 0.985,
        outsideFeatherScale: 0.004
    )

    static let uvSegProduct = UVLipEdgeRefinement(
        radius: 2,
        insetPixels: 0.42,
        featherPixels: 0.48,
        hardThreshold: 104,
        maximumAlpha: 255,
        insideRetain: 0.988,
        outsideFeatherScale: 0.004
    )

    func normalizedAlpha(signedDistance: Float) -> Float {
        guard featherPixels > 0 else {
            return signedDistance > insetPixels ? 1 : 0
        }

        let shiftedDistance = signedDistance - insetPixels
        let rawT = (shiftedDistance + featherPixels) / (featherPixels * 2)
        let t = min(max(rawT, 0), 1)
        let smoothT = t * t * (3 - (2 * t))
        if signedDistance < 0 {
            return min(max(smoothT * outsideFeatherScale, 0), outsideFeatherScale)
        }
        return smoothT
    }
}

final class UVLipMaskMetalFeather: @unchecked Sendable {
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void featherLipAlpha(
        const device uchar *input [[buffer(0)]],
        device uchar *output [[buffer(1)]],
        constant uint &width [[buffer(2)]],
        constant uint &height [[buffer(3)]],
        constant uint &radius [[buffer(4)]],
        constant float &edgeWidth [[buffer(5)]],
        constant uchar &hardThreshold [[buffer(6)]],
        constant float &insetPixels [[buffer(7)]],
        constant float &maxAlpha [[buffer(8)]],
        constant float &insideRetain [[buffer(9)]],
        constant float &outsideFeatherScale [[buffer(10)]],
        uint gid [[thread_position_in_grid]]
    ) {
        const uint count = width * height;
        if (gid >= count) {
            return;
        }

        const uint x = gid % width;
        const uint y = gid / width;
        const uchar center = input[gid];
        const bool inside = center >= hardThreshold;
        float bestDistanceSquared = 9999.0;

        const int searchRadius = int(radius);
        for (int dy = -searchRadius; dy <= searchRadius; dy++) {
            const int yy = int(y) + dy;
            if (yy < 0 || yy >= int(height)) {
                continue;
            }

            for (int dx = -searchRadius; dx <= searchRadius; dx++) {
                const int xx = int(x) + dx;
                if (xx < 0 || xx >= int(width)) {
                    continue;
                }

                const uint index = uint(yy) * width + uint(xx);
                const bool neighborInside = input[index] >= hardThreshold;
                if (neighborInside == inside) {
                    continue;
                }

                const float distanceSquared = float((dx * dx) + (dy * dy));
                bestDistanceSquared = min(bestDistanceSquared, distanceSquared);
            }
        }

        if (bestDistanceSquared > 9000.0) {
            output[gid] = inside ? uchar(clamp(max(float(center) * insideRetain, maxAlpha * 0.86), 0.0, maxAlpha)) : uchar(0);
            return;
        }

        const float distance = sqrt(bestDistanceSquared);
        const float signedDistance = inside ? distance : -distance;
        const float shiftedDistance = signedDistance - insetPixels;
        const float t = clamp((shiftedDistance + edgeWidth) / (edgeWidth * 2.0), 0.0, 1.0);
        const float smoothT = t * t * (3.0 - (2.0 * t));
        const float sdfAlpha = smoothT * maxAlpha;

        if (inside) {
            const float retained = float(center) * insideRetain;
            output[gid] = uchar(clamp(max(retained, sdfAlpha), 0.0, maxAlpha));
        } else {
            const float outsideCap = maxAlpha * outsideFeatherScale;
            const float retained = float(center) * 0.08;
            output[gid] = uchar(clamp(max(retained, sdfAlpha * outsideFeatherScale), 0.0, outsideCap));
        }
    }
    """

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let pipelineState: MTLComputePipelineState?

    init() {
        let metalDevice = MTLCreateSystemDefaultDevice()
        device = metalDevice
        commandQueue = metalDevice?.makeCommandQueue()

        if
            let metalDevice,
            let library = try? metalDevice.makeLibrary(source: Self.shaderSource, options: nil),
            let function = library.makeFunction(name: "featherLipAlpha")
        {
            pipelineState = try? metalDevice.makeComputePipelineState(function: function)
        } else {
            pipelineState = nil
        }
    }

    func feathered(
        _ mask: UVLipAlphaMask,
        refinement: UVLipEdgeRefinement = .product
    ) -> UVLipAlphaMask? {
        let count = mask.width * mask.height
        guard
            count > 0,
            mask.alpha.count == count,
            let device,
            let commandQueue,
            let pipelineState,
            let inputBuffer = mask.alpha.withUnsafeBytes({ rawBuffer in
                rawBuffer.baseAddress.flatMap {
                    device.makeBuffer(bytes: $0, length: count, options: .storageModeShared)
                }
            }),
            let outputBuffer = device.makeBuffer(length: count, options: .storageModeShared),
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }

        var width = UInt32(mask.width)
        var height = UInt32(mask.height)
        var radius = refinement.radius
        var edgeWidth = refinement.featherPixels
        var hardThreshold = refinement.hardThreshold
        var insetPixels = refinement.insetPixels
        var maximumAlpha = refinement.maximumAlpha
        var insideRetain = refinement.insideRetain
        var outsideFeatherScale = refinement.outsideFeatherScale

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&height, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&radius, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&edgeWidth, length: MemoryLayout<Float>.size, index: 5)
        encoder.setBytes(&hardThreshold, length: MemoryLayout<UInt8>.size, index: 6)
        encoder.setBytes(&insetPixels, length: MemoryLayout<Float>.size, index: 7)
        encoder.setBytes(&maximumAlpha, length: MemoryLayout<Float>.size, index: 8)
        encoder.setBytes(&insideRetain, length: MemoryLayout<Float>.size, index: 9)
        encoder.setBytes(&outsideFeatherScale, length: MemoryLayout<Float>.size, index: 10)

        let threads = min(max(pipelineState.threadExecutionWidth, 1), pipelineState.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        guard commandBuffer.status == .completed else { return nil }

        let pointer = outputBuffer.contents().bindMemory(to: UInt8.self, capacity: count)
        let alpha = Array(UnsafeBufferPointer(start: pointer, count: count))
        return UVLipAlphaMask(width: mask.width, height: mask.height, alpha: alpha)
    }
}

enum UVMakeupTexture {
    static func render(
        look: MakeupLook,
        intensity: Double,
        faceGeometry: ARFaceGeometry?,
        faceParsingMask: UVFaceParsingMask? = nil,
        lipAlphaMask: UVLipAlphaMask? = nil,
        lipLineMask: UVLipAlphaMask? = nil,
        drawsLips: Bool = true,
        drawsFaceMakeup: Bool = true,
        showsDebugGuides: Bool = false,
        preserveClosedMouthCrease: Bool = true,
        prefersTightLipMask: Bool = false,
        fastLipOnlyPreview: Bool = false
    ) -> UIImage {
        let size = CGSize(width: 1024, height: 1024)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false

        return UIGraphicsImageRenderer(size: size, format: format).image { rendererContext in
            let context = rendererContext.cgContext
            let rect = CGRect(origin: .zero, size: size)
            context.setShouldAntialias(true)
            context.interpolationQuality = .high

            if drawsFaceMakeup {
                let skinAlphaMask = faceParsingMask?.alphaMask(
                    for: [.skin, .nose],
                    minimumConfidence: 18,
                    minimumCoverage: 0.08,
                    maximumCoverage: 0.58,
                    expansionRadius: 2,
                    blurRadius: 2
                )
                drawFoundation(look.foundation.uiColor(intensity: intensity), alphaMask: skinAlphaMask, context: context, rect: rect)
                drawContour(intensity: intensity, context: context, rect: rect)
                drawCheek(.left, color: look.blush.uiColor(intensity: intensity), context: context, rect: rect)
                drawCheek(.right, color: look.blush.uiColor(intensity: intensity), context: context, rect: rect)
                drawHighlight(intensity: intensity, context: context, rect: rect)
                drawBrow(.left, color: look.eye.uiColor(intensity: intensity), context: context, rect: rect)
                drawBrow(.right, color: look.eye.uiColor(intensity: intensity), context: context, rect: rect)
                drawEyeShadow(.left, color: look.eye.uiColor(intensity: intensity), context: context, rect: rect)
                drawEyeShadow(.right, color: look.eye.uiColor(intensity: intensity), context: context, rect: rect)
            }
            if drawsLips {
                let parsingLipAlphaMask = faceParsingMask?.lipstickAlphaMask()
                let dynamicLipMask = faceGeometry.flatMap { DynamicLipMask(faceGeometry: $0) }
                if look.isLipLineDebug {
                    drawLipLineDebug(
                        lipLineMask,
                        context: context,
                        rect: rect
                    )
                } else {
                    drawLips(
                        look.lip.uiColor(intensity: intensity),
                        mask: dynamicLipMask,
                        alphaMask: lipAlphaMask ?? parsingLipAlphaMask,
                        lipLineMask: lipLineMask,
                        preserveClosedMouthCrease: preserveClosedMouthCrease,
                        prefersTightLipMask: prefersTightLipMask,
                        fastLipOnlyPreview: fastLipOnlyPreview,
                        showsDebugGuides: showsDebugGuides,
                        context: context,
                        rect: rect
                    )
                }
            }
        }
    }

    static func renderFastLipOnlyPreview(
        look: MakeupLook,
        intensity: Double,
        lipAlphaMask: UVLipAlphaMask,
        preserveClosedMouthCrease: Bool
    ) -> UIImage {
        let size = CGSize(width: 1024, height: 1024)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false

        return UIGraphicsImageRenderer(size: size, format: format).image { rendererContext in
            let context = rendererContext.cgContext
            let rect = CGRect(origin: .zero, size: size)
            let baseColor = look.lip.uiColor(intensity: intensity)
                .withAlphaComponent(min(max(look.lip.uiColor(intensity: intensity).alpha * 0.96, 0.08), 0.88))
            let baseMask = preserveClosedMouthCrease
                ? lipAlphaMask.preservedClosedMouthCrease()
                : lipAlphaMask
            let previewMask = baseMask.productContourSmoothedForMakeup(
                edgeThreshold: 44,
                smoothingIterations: 2,
                edgeFeatherPixels: 0.56,
                outputScale: 1
            )

            context.setFillColor(baseColor.cgColor)
            context.fill(rect)
            apply(alphaMask: previewMask, context: context, rect: rect)
        }
    }

    static func renderUltraFastLipOnlyPreview(
        look: MakeupLook,
        intensity: Double,
        lipAlphaMask: UVLipAlphaMask,
        preserveClosedMouthCrease: Bool
    ) -> UIImage {
        let size = CGSize(width: 1024, height: 1024)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false

        return UIGraphicsImageRenderer(size: size, format: format).image { rendererContext in
            let context = rendererContext.cgContext
            let rect = CGRect(origin: .zero, size: size)
            let baseColor = look.lip.uiColor(intensity: intensity)
                .withAlphaComponent(min(max(look.lip.uiColor(intensity: intensity).alpha * 0.90, 0.06), 0.84))
            let previewMask = preserveClosedMouthCrease
                ? lipAlphaMask.preservedClosedMouthCrease()
                : lipAlphaMask

            context.setFillColor(baseColor.cgColor)
            context.fill(rect)
            apply(alphaMask: previewMask, context: context, rect: rect)
        }
    }

    private static func refinedLipOnlyPreviewMask(
        previewMask: UVLipAlphaMask,
        sourceMask: UVLipAlphaMask
    ) -> UVLipAlphaMask {
        guard let styleMask = lipStyleClipMask(for: sourceMask) else {
            return previewMask
        }
        return previewMask
            .clipped(to: styleMask)
            .softenedForMakeup(edgeThreshold: 16, innerBlurRadius: 1, outerFeatherRadius: 3)
    }

    private static func lipStyleClipMask(for alphaMask: UVLipAlphaMask) -> UVLipAlphaMask? {
        let lipSafeBounds = UVBounds(minU: 0.388, maxU: 0.612, minV: 0.582, maxV: 0.722)
        guard
            let rawBounds = activeBounds(for: alphaMask, threshold: 18),
            rawBounds.isValid
        else {
            return nil
        }

        let mouthBounds = rawBounds
            .expanded(x: 0.025, y: 0.10)
            .clamped(to: lipSafeBounds)
        guard mouthBounds.isValid else { return nil }

        let upperBottom = mouthBounds.minV + mouthBounds.height * 0.45
        let lowerTop = mouthBounds.minV + mouthBounds.height * 0.39
        let upperBounds = UVBounds(
            minU: mouthBounds.minU + mouthBounds.width * 0.01,
            maxU: mouthBounds.maxU - mouthBounds.width * 0.01,
            minV: mouthBounds.minV,
            maxV: min(upperBottom, mouthBounds.maxV)
        )
        let lowerBounds = UVBounds(
            minU: mouthBounds.minU,
            maxU: mouthBounds.maxU,
            minV: max(lowerTop, mouthBounds.minV),
            maxV: mouthBounds.maxV
        )
        guard upperBounds.isValid, lowerBounds.isValid else { return nil }

        return rasterizedLipPathMask(
            width: alphaMask.width,
            height: alphaMask.height,
            upperLip: UVPath.lipUpper(in: upperBounds),
            lowerLip: UVPath.lipLower(in: lowerBounds)
        )?
        .softenedForMakeup(edgeThreshold: 12, innerBlurRadius: 0, outerFeatherRadius: 2)
    }

    private static func activeBounds(for alphaMask: UVLipAlphaMask, threshold: UInt8) -> UVBounds? {
        guard
            alphaMask.width > 1,
            alphaMask.height > 1,
            alphaMask.alpha.count == alphaMask.width * alphaMask.height
        else {
            return nil
        }

        var minX = alphaMask.width
        var maxX = -1
        var minY = alphaMask.height
        var maxY = -1
        for y in 0..<alphaMask.height {
            let rowOffset = y * alphaMask.width
            for x in 0..<alphaMask.width where alphaMask.alpha[rowOffset + x] > threshold {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }
        guard minX <= maxX, minY <= maxY else { return nil }

        let widthScale = CGFloat(max(alphaMask.width - 1, 1))
        let heightScale = CGFloat(max(alphaMask.height - 1, 1))
        return UVBounds(
            minU: CGFloat(minX) / widthScale,
            maxU: CGFloat(maxX) / widthScale,
            minV: CGFloat(minY) / heightScale,
            maxV: CGFloat(maxY) / heightScale
        )
    }

    private static func rasterizedLipPathMask(
        width: Int,
        height: Int,
        upperLip: UVPath,
        lowerLip: UVPath
    ) -> UVLipAlphaMask? {
        guard
            width > 2,
            height > 2,
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
        else {
            return nil
        }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(rect)
        context.setFillColor(gray: 1, alpha: 1)
        context.addPath(upperLip.path(in: rect).cgPath)
        context.fillPath()
        context.addPath(lowerLip.path(in: rect).cgPath)
        context.fillPath()

        guard let data = context.data else { return nil }
        let pointer = data.bindMemory(to: UInt8.self, capacity: width * height)
        let alpha = Array(UnsafeBufferPointer(start: pointer, count: width * height))
            .map { UInt8(min(Int($0) * 255, 255)) }
        return UVLipAlphaMask(width: width, height: height, alpha: alpha)
    }

    private static func drawFoundation(
        _ color: UIColor,
        alphaMask: UVLipAlphaMask?,
        context: CGContext,
        rect: CGRect
    ) {
        let path = UVMakeupLandmarks.faceOval.path(in: rect)

        context.saveGState()
        context.beginTransparencyLayer(auxiliaryInfo: nil)
        fill(path, color: color.withAlphaComponent(color.alpha * 0.72), context: context, feather: 32)
        apply(alphaMask: alphaMask, context: context, rect: rect)
        context.endTransparencyLayer()
        context.restoreGState()
    }

    private static func drawContour(intensity: Double, context: CGContext, rect: CGRect) {
        let color = UIColor(red: 0.22, green: 0.12, blue: 0.10, alpha: 0.16 * CGFloat(intensity))
        fill(UVMakeupLandmarks.noseContour(.left).path(in: rect), color: color, context: context, feather: 18)
        fill(UVMakeupLandmarks.noseContour(.right).path(in: rect), color: color, context: context, feather: 18)
        fill(UVMakeupLandmarks.cheekContour(.left).path(in: rect), color: color.withAlphaComponent(color.alpha * 0.58), context: context, feather: 28)
        fill(UVMakeupLandmarks.cheekContour(.right).path(in: rect), color: color.withAlphaComponent(color.alpha * 0.58), context: context, feather: 28)
    }

    private static func drawHighlight(intensity: Double, context: CGContext, rect: CGRect) {
        let color = UIColor(red: 1.0, green: 0.86, blue: 0.72, alpha: 0.13 * CGFloat(intensity))
        drawRadialPatch(
            center: UVPoint(0.500, 0.415).point(in: rect),
            radius: CGSize(width: rect.width * 0.026, height: rect.height * 0.126),
            color: color,
            context: context,
            alphaStops: [0.86, 0.28, 0]
        )
        drawRadialPatch(
            center: UVPoint(0.385, 0.474).point(in: rect),
            radius: CGSize(width: rect.width * 0.080, height: rect.height * 0.040),
            color: color.withAlphaComponent(color.alpha * 0.72),
            context: context,
            alphaStops: [0.82, 0.22, 0]
        )
        drawRadialPatch(
            center: UVPoint(0.615, 0.474).point(in: rect),
            radius: CGSize(width: rect.width * 0.080, height: rect.height * 0.040),
            color: color.withAlphaComponent(color.alpha * 0.72),
            context: context,
            alphaStops: [0.82, 0.22, 0]
        )
    }

    private static func drawBrow(
        _ side: FaceSide,
        color: UIColor,
        context: CGContext,
        rect: CGRect
    ) {
        let browColor = UIColor(
            red: 0.16,
            green: 0.09,
            blue: 0.08,
            alpha: min(max(color.alpha * 0.42, 0), 0.18)
        )
        fill(UVMakeupLandmarks.brow(side).path(in: rect), color: browColor, context: context, feather: 7)
    }

    private static func drawCheek(
        _ side: FaceSide,
        color: UIColor,
        context: CGContext,
        rect: CGRect
    ) {
        let center = UVMakeupLandmarks.cheekCenter(side).point(in: rect)
        let radius = CGSize(width: rect.width * 0.108, height: rect.height * 0.078)
        drawRadialPatch(
            center: center,
            radius: radius,
            color: color,
            context: context,
            alphaStops: [0.96, 0.42, 0]
        )
    }

    private static func drawEyeShadow(
        _ side: FaceSide,
        color: UIColor,
        context: CGContext,
        rect: CGRect
    ) {
        let path = UVMakeupLandmarks.eyeShadow(side).path(in: rect)
        fill(path, color: color.withAlphaComponent(color.alpha * 0.86), context: context, feather: 20)

        let center = UVMakeupLandmarks.eyeCenter(side).point(in: rect)
        drawRadialPatch(
            center: center,
            radius: CGSize(width: rect.width * 0.070, height: rect.height * 0.032),
            color: color.withAlphaComponent(color.alpha * 0.52),
            context: context,
            alphaStops: [0.74, 0.24, 0]
        )
    }

    private static func drawLips(
        _ color: UIColor,
        mask: DynamicLipMask?,
        alphaMask: UVLipAlphaMask?,
        lipLineMask: UVLipAlphaMask?,
        preserveClosedMouthCrease: Bool,
        prefersTightLipMask: Bool,
        fastLipOnlyPreview: Bool,
        showsDebugGuides: Bool,
        context: CGContext,
        rect: CGRect
    ) {
        let lipMask = mask
        let upperLipPath = lipMask?.upperLip ?? UVMakeupLandmarks.upperLip
        let lowerLipPath = lipMask?.lowerLip ?? UVMakeupLandmarks.lowerLip
        if let alphaMask {
            drawMaskedLips(
                color,
                alphaMask: alphaMask,
                lipLineMask: lipLineMask,
                paintClipPath: nil,
                creasePath: lipMask?.crease ?? UVMakeupLandmarks.lipCrease,
                highlightPath: lipMask?.highlight ?? UVMakeupLandmarks.lipHighlight,
                preserveClosedMouthCrease: preserveClosedMouthCrease,
                prefersTightLipMask: prefersTightLipMask,
                fastLipOnlyPreview: fastLipOnlyPreview,
                showsDebugGuides: showsDebugGuides,
                context: context,
                rect: rect
            )
            return
        }

        let upperLip = upperLipPath.path(in: rect)
        let lowerLip = lowerLipPath.path(in: rect)
        guard color.alpha > 0.001 else {
            if showsDebugGuides {
                let crease = (lipMask?.crease ?? UVMakeupLandmarks.lipCrease).path(in: rect)
                stroke(upperLip, color: UIColor.systemCyan.withAlphaComponent(0.82), context: context, lineWidth: 5)
                stroke(lowerLip, color: UIColor.systemYellow.withAlphaComponent(0.82), context: context, lineWidth: 5)
                stroke(crease, color: UIColor.white.withAlphaComponent(0.78), context: context, lineWidth: 4)
            }
            return
        }

        context.saveGState()
        context.beginTransparencyLayer(auxiliaryInfo: nil)

        fill(upperLip, color: color.withAlphaComponent(color.alpha * 0.16), context: context, feather: 22)
        fill(lowerLip, color: color.withAlphaComponent(color.alpha * 0.20), context: context, feather: 24)

        let innerUpperLip = upperLipPath.scaled(x: 0.88, y: 0.72).path(in: rect)
        let innerLowerLip = lowerLipPath.scaled(x: 0.90, y: 0.80).path(in: rect)
        fill(innerUpperLip, color: color.withAlphaComponent(color.alpha * 0.46), context: context, feather: 10)
        fill(innerLowerLip, color: color.withAlphaComponent(color.alpha * 0.52), context: context, feather: 10)

        let lowerSaturation = lowerLipPath.scaled(x: 0.78, y: 0.58).path(in: rect)
        fill(lowerSaturation, color: color.withAlphaComponent(color.alpha * 0.14), context: context, feather: 16)

        let crease = (lipMask?.crease ?? UVMakeupLandmarks.lipCrease).path(in: rect)
        stroke(crease, color: UIColor.black.withAlphaComponent(max(color.alpha * 0.045, 0.020)), context: context, lineWidth: 2)

        apply(alphaMask: alphaMask, context: context, rect: rect)

        context.endTransparencyLayer()
        context.restoreGState()

        if showsDebugGuides {
            stroke(upperLip, color: UIColor.systemCyan.withAlphaComponent(0.82), context: context, lineWidth: 5)
            stroke(lowerLip, color: UIColor.systemYellow.withAlphaComponent(0.82), context: context, lineWidth: 5)
            stroke(crease, color: UIColor.white.withAlphaComponent(0.78), context: context, lineWidth: 4)
        }
    }

    private static func drawMaskedLips(
        _ color: UIColor,
        alphaMask: UVLipAlphaMask,
        lipLineMask: UVLipAlphaMask?,
        paintClipPath: UVPath?,
        creasePath: UVPath,
        highlightPath: UVPath,
        preserveClosedMouthCrease: Bool,
        prefersTightLipMask: Bool,
        fastLipOnlyPreview: Bool,
        showsDebugGuides: Bool,
        context: CGContext,
        rect: CGRect
    ) {
        guard color.alpha > 0.001 else { return }

        context.saveGState()
        context.beginTransparencyLayer(auxiliaryInfo: nil)
        if let paintClipPath {
            context.addPath(paintClipPath.path(in: rect).cgPath)
            context.clip()
        }

        let usesCoverageDebugColor = showsDebugGuides || isCoverageDebugColor(color)
        let usesProjectedLipRefinement = lipLineMask != nil && !usesCoverageDebugColor
        let baseColor = usesCoverageDebugColor
            ? color.withAlphaComponent(min(max(color.alpha * 0.92, 0.72), 0.96))
            : fastLipOnlyPreview
                ? color.withAlphaComponent(min(max(color.alpha * 0.84, 0.54), 0.90))
            : usesProjectedLipRefinement
                ? color.withAlphaComponent(min(max(color.alpha * 0.86, 0.58), 0.88))
            : color.withAlphaComponent(min(max(color.alpha * 0.78, 0.46), 0.82))
        let renderAlphaMask: UVLipAlphaMask
        if usesCoverageDebugColor {
            // Coverage debug should reflect the incoming UV mask as directly as possible.
            let baseMask = preserveClosedMouthCrease
                ? alphaMask.preservedClosedMouthCrease()
                : alphaMask
            renderAlphaMask = baseMask.expanded(
                threshold: 24,
                radius: 0,
                blurRadius: 1
            )
        } else if usesProjectedLipRefinement {
            let baseMask = preserveClosedMouthCrease
                ? alphaMask.preservedClosedMouthCrease()
                : alphaMask
            renderAlphaMask = baseMask.productRenderMaskForMakeup(
                targetSize: Int(rect.width),
                preserveClosedMouthCrease: false
            )
        } else if prefersTightLipMask {
            let baseMask = preserveClosedMouthCrease
                ? alphaMask.preservedClosedMouthCrease()
                : alphaMask
            renderAlphaMask = baseMask.productContourSmoothedForMakeup(
                edgeThreshold: 48,
                smoothingIterations: 2,
                edgeFeatherPixels: 0.72,
                outputScale: 3
            )
        } else {
            renderAlphaMask = alphaMask.productRenderMaskForMakeup(
                targetSize: Int(rect.width),
                preserveClosedMouthCrease: preserveClosedMouthCrease
            )
        }
        context.setFillColor(baseColor.cgColor)
        context.fill(rect)

        if !usesCoverageDebugColor && !usesProjectedLipRefinement && !fastLipOnlyPreview {
            drawLipMaterialBloom(color, context: context, rect: rect)

            context.setBlendMode(.multiply)
            stroke(
                creasePath.path(in: rect),
                color: UIColor.black.withAlphaComponent(min(max(color.alpha * 0.055, 0.020), 0.055)),
                context: context,
                lineWidth: 2
            )

            context.setBlendMode(.screen)
            stroke(
                highlightPath.path(in: rect),
                color: UIColor.white.withAlphaComponent(min(max(color.alpha * 0.055, 0.018), 0.052)),
                context: context,
                lineWidth: 5
            )
            context.setBlendMode(.normal)
        }

        apply(alphaMask: renderAlphaMask, context: context, rect: rect)

        if usesProjectedLipRefinement {
            context.setBlendMode(.multiply)
            stroke(
                creasePath.path(in: rect),
                color: UIColor.black.withAlphaComponent(min(max(color.alpha * 0.12, 0.05), 0.18)),
                context: context,
                lineWidth: 2
            )
            context.setBlendMode(.normal)
        }

        if let lipLineMask {
            drawLipLineOverlay(
                color,
                lineMask: lipLineMask,
                usesCoverageDebugColor: usesCoverageDebugColor,
                context: context,
                rect: rect
            )
        }

        context.endTransparencyLayer()
        context.restoreGState()
    }

    private static func drawLipLineDebug(
        _ alphaMask: UVLipAlphaMask?,
        context: CGContext,
        rect: CGRect
    ) {
        guard let alphaMask else { return }

        let outlineMask = alphaMask.lipLineDebugOutline(threshold: 88, widthPixels: 1)

        context.saveGState()
        context.beginTransparencyLayer(auxiliaryInfo: nil)
        context.setFillColor(UIColor(red: 0.00, green: 1.00, blue: 0.18, alpha: 0.96).cgColor)
        context.fill(rect)
        apply(alphaMask: outlineMask, context: context, rect: rect)
        context.endTransparencyLayer()
        context.restoreGState()
    }

    private static func drawLipLineOverlay(
        _ color: UIColor,
        lineMask: UVLipAlphaMask,
        usesCoverageDebugColor: Bool,
        context: CGContext,
        rect: CGRect
    ) {
        let renderMask = usesCoverageDebugColor
            ? lineMask.softenedStroke(expansionRadius: 0, blurRadius: 0)
            : lineMask
                .softenedStroke(expansionRadius: 0, blurRadius: 1)
                .productContourSmoothedForMakeup(
                    edgeThreshold: 32,
                    smoothingIterations: 1,
                    edgeFeatherPixels: 0.45,
                    outputScale: 2
                )
        let lineColor = lipLineColor(from: color, usesCoverageDebugColor: usesCoverageDebugColor)
        context.saveGState()
        context.beginTransparencyLayer(auxiliaryInfo: nil)
        context.setFillColor(lineColor.cgColor)
        context.fill(rect)
        apply(alphaMask: renderMask, context: context, rect: rect)
        context.endTransparencyLayer()
        context.restoreGState()
    }

    private static func isCoverageDebugColor(_ color: UIColor) -> Bool {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard color.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else { return false }
        return green > 0.80 && red < 0.16 && blue < 0.32 && alpha > 0.50
    }

    private static func lipLineColor(from color: UIColor, usesCoverageDebugColor: Bool) -> UIColor {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard color.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return color
        }

        let darken: CGFloat = usesCoverageDebugColor ? 0.10 : 0.36
        let opacity = usesCoverageDebugColor
            ? min(max(alpha * 0.96, 0.84), 0.98)
            : min(max(alpha * 0.92, 0.48), 0.92)
        return UIColor(
            red: red * (1 - darken),
            green: green * (1 - darken),
            blue: blue * (1 - darken),
            alpha: opacity
        )
    }

    private static func drawLipMaterialBloom(
        _ color: UIColor,
        context: CGContext,
        rect: CGRect
    ) {
        context.saveGState()
        context.setBlendMode(.normal)
        drawRadialPatch(
            center: UVPoint(0.500, 0.645).point(in: rect),
            radius: CGSize(width: rect.width * 0.090, height: rect.height * 0.042),
            color: color.withAlphaComponent(min(max(color.alpha * 0.11, 0.035), 0.10)),
            context: context,
            alphaStops: [0.92, 0.36, 0]
        )

        context.setBlendMode(.screen)
        drawRadialPatch(
            center: UVPoint(0.500, 0.675).point(in: rect),
            radius: CGSize(width: rect.width * 0.070, height: rect.height * 0.028),
            color: UIColor.white.withAlphaComponent(min(max(color.alpha * 0.045, 0.012), 0.036)),
            context: context,
            alphaStops: [0.86, 0.30, 0]
        )
        context.restoreGState()
    }

    private static func drawGlossStreaks(alpha: CGFloat, context: CGContext, rect: CGRect) {
        let glossAlpha = min(max(alpha * 0.10, 0.035), 0.085)
        let streaks: [(CGFloat, CGFloat, CGFloat)] = [
            (0.465, 0.646, 0.030),
            (0.488, 0.651, 0.040),
            (0.512, 0.651, 0.038),
            (0.535, 0.646, 0.030)
        ]

        for (u, v, length) in streaks {
            let start = UVPoint(u, v).point(in: rect)
            let end = UVPoint(u + 0.004, v + length).point(in: rect)
            let control = UVPoint(u - 0.006, v + length * 0.48).point(in: rect)
            let path = UIBezierPath()
            path.move(to: start)
            path.addQuadCurve(to: end, controlPoint: control)
            stroke(path, color: UIColor.white.withAlphaComponent(glossAlpha), context: context, lineWidth: 3)
        }
    }

    private static func apply(alphaMask: UVLipAlphaMask?, context: CGContext, rect: CGRect) {
        guard let image = alphaMask?.cgImage() else { return }

        context.saveGState()
        context.interpolationQuality = .high
        UIImage(cgImage: image).draw(in: rect, blendMode: .destinationIn, alpha: 1)
        context.restoreGState()
    }

    private static func fill(
        _ path: UIBezierPath,
        color: UIColor,
        context: CGContext,
        feather: CGFloat
    ) {
        context.saveGState()
        context.addPath(path.cgPath)
        context.setShadow(offset: .zero, blur: feather, color: color.cgColor)
        color.setFill()
        context.fillPath()
        context.restoreGState()

        color.setFill()
        path.fill()
    }

    private static func stroke(
        _ path: UIBezierPath,
        color: UIColor,
        context: CGContext,
        lineWidth: CGFloat
    ) {
        context.saveGState()
        context.addPath(path.cgPath)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.setLineWidth(lineWidth)
        context.setStrokeColor(color.cgColor)
        context.strokePath()
        context.restoreGState()
    }

    private static func drawRadialPatch(
        center: CGPoint,
        radius: CGSize,
        color: UIColor,
        context: CGContext,
        alphaStops: [CGFloat]
    ) {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let colors = alphaStops.map { color.withAlphaComponent(color.alpha * $0).cgColor } as CFArray
        let locations: [CGFloat] = [0, 0.48, 1]
        guard let gradient = CGGradient(colorsSpace: colorSpace, colors: colors, locations: locations) else {
            return
        }

        context.saveGState()
        context.translateBy(x: center.x, y: center.y)
        context.scaleBy(x: radius.width / radius.height, y: 1)
        context.drawRadialGradient(
            gradient,
            startCenter: .zero,
            startRadius: 1,
            endCenter: .zero,
            endRadius: radius.height,
            options: [.drawsAfterEndLocation]
        )
        context.restoreGState()
    }
}

private struct DynamicLipMask {
    let upperLip: UVPath
    let lowerLip: UVPath
    let crease: UVPath
    let highlight: UVPath
    let lipPaintClip: UVPath
    let safetyClip: UVPath

    nonisolated init?(faceGeometry: ARFaceGeometry) {
        let vertices = faceGeometry.vertices
        let textureCoordinates = faceGeometry.textureCoordinates
        guard vertices.count == textureCoordinates.count, vertices.count > 100 else {
            return nil
        }

        let bounds = FaceBounds(vertices: vertices)
        let candidates = vertices.indices.compactMap { index -> VertexSample? in
            let vertex = vertices[index]
            let normalized = bounds.normalized(vertex)

            guard
                normalized.x > 0.30,
                normalized.x < 0.70,
                normalized.yDown > 0.58,
                normalized.yDown < 0.77
            else {
                return nil
            }

            return VertexSample(
                facePoint: normalized,
                texturePoint: UVPoint(textureCoordinates[index])
            )
        }

        guard candidates.count >= 12 else {
            return nil
        }

        let lipSafeBounds = UVBounds(minU: 0.392, maxU: 0.608, minV: 0.586, maxV: 0.716)
        let mouthBounds = UVBounds(points: candidates.map(\.texturePoint))
            .expanded(x: 0.03, y: 0.035)
            .clamped()
            .clamped(to: lipSafeBounds)
        let safetyBounds = mouthBounds
            .expanded(x: 0.20, y: 0.32)
            .clamped(to: lipSafeBounds)
        let paintBounds = mouthBounds
            .expanded(x: 0.04, y: 0.08)
            .clamped(to: lipSafeBounds)

        let midY = mouthBounds.midY
        let upperBounds = UVBounds(
            points: candidates
                .map(\.texturePoint)
                .filter { $0.v <= midY }
        )
        .expanded(x: 0.015, y: 0.040)
        .clamped(to: mouthBounds)

        let lowerBounds = UVBounds(
            points: candidates
                .map(\.texturePoint)
                .filter { $0.v > midY }
        )
        .expanded(x: 0.020, y: 0.070)
        .clamped(to: mouthBounds)

        guard upperBounds.isValid, lowerBounds.isValid else {
            return nil
        }

        upperLip = UVPath.lipUpper(in: upperBounds)
        lowerLip = UVPath.lipLower(in: lowerBounds)
        crease = UVPath.lipCrease(in: mouthBounds)
        highlight = UVPath.lipHighlight(in: lowerBounds)
        lipPaintClip = UVPath.mouthSafetyClip(in: paintBounds)
        safetyClip = UVPath.mouthSafetyClip(in: safetyBounds)
    }
}

private extension UVFaceParsingMask {
    func lipstickAlphaMask() -> UVLipAlphaMask? {
        guard
            let rawLipMask = alphaMask(
                for: [.upperLip, .lowerLip],
                minimumConfidence: 10,
                minimumCoverage: 0.0018,
                maximumCoverage: 0.160,
                expansionRadius: 2,
                blurRadius: 1
            )
        else {
            return nil
        }

        var alpha = rawLipMask.alpha

        if let innerMouthMask = alphaMask(
            for: [.innerMouth],
            minimumConfidence: 12,
            minimumCoverage: 0.0001,
            maximumCoverage: 0.070,
            expansionRadius: 2,
            blurRadius: 2
        ) {
            alpha = subtract(innerMouthMask.alpha, from: alpha)
        }

        alpha = removeTinyComponents(
            from: alpha,
            width: width,
            height: height,
            threshold: 20,
            minimumPixels: max(18, Int(CGFloat(width * height) * 0.00034))
        )
        alpha = closeHorizontalGaps(alpha, width: width, height: height, radius: 2)
        alpha = blurred(alpha: alpha, width: width, height: height, radius: 1)

        let activePixelCount = alpha.reduce(0) { $0 + ($1 > 16 ? 1 : 0) }
        let coverage = CGFloat(activePixelCount) / CGFloat(max(width * height, 1))
        guard coverage >= 0.0015, coverage <= 0.16 else { return nil }

        return UVLipAlphaMask(width: width, height: height, alpha: alpha)
    }

    private func subtract(_ cutout: [UInt8], from source: [UInt8]) -> [UInt8] {
        guard cutout.count == source.count else { return source }

        var output = source
        for index in output.indices where cutout[index] > 0 {
            let reduced = Int(output[index]) - Int(cutout[index]) * 2
            output[index] = UInt8(min(max(reduced, 0), 255))
        }
        return output
    }

    private func removeTinyComponents(
        from alpha: [UInt8],
        width: Int,
        height: Int,
        threshold: UInt8,
        minimumPixels: Int
    ) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else { return alpha }

        var output = alpha
        var visited = [Bool](repeating: false, count: alpha.count)
        var stack: [Int] = []
        var component: [Int] = []

        for start in alpha.indices where !visited[start] && alpha[start] > threshold {
            stack.removeAll(keepingCapacity: true)
            component.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true

            while let index = stack.popLast() {
                component.append(index)
                let x = index % width
                let y = index / width

                for neighbor in neighbors(x: x, y: y, width: width, height: height) where !visited[neighbor] {
                    visited[neighbor] = true
                    if alpha[neighbor] > threshold {
                        stack.append(neighbor)
                    }
                }
            }

            if component.count < minimumPixels {
                for index in component {
                    output[index] = 0
                }
            }
        }

        return output
    }

    private func neighbors(x: Int, y: Int, width: Int, height: Int) -> [Int] {
        var result: [Int] = []
        if x > 0 { result.append(y * width + x - 1) }
        if x < width - 1 { result.append(y * width + x + 1) }
        if y > 0 { result.append((y - 1) * width + x) }
        if y < height - 1 { result.append((y + 1) * width + x) }
        return result
    }

    private func closeHorizontalGaps(_ alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else { return alpha }

        var output = alpha
        for y in 0..<height {
            for x in radius..<(width - radius) where alpha[y * width + x] == 0 {
                let left = alpha[y * width + x - radius]
                let right = alpha[y * width + x + radius]
                guard left > 42, right > 42 else { continue }
                output[y * width + x] = UInt8((Int(left) + Int(right)) / 3)
            }
        }
        return output
    }

    private func blurred(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else { return alpha }

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

private struct FaceBounds {
    let minX: Float
    let maxX: Float
    let minY: Float
    let maxY: Float

    nonisolated init(vertices: [simd_float3]) {
        minX = vertices.map(\.x).min() ?? -1
        maxX = vertices.map(\.x).max() ?? 1
        minY = vertices.map(\.y).min() ?? -1
        maxY = vertices.map(\.y).max() ?? 1
    }

    nonisolated func normalized(_ vertex: simd_float3) -> NormalizedFacePoint {
        NormalizedFacePoint(
            x: CGFloat((vertex.x - minX) / max(maxX - minX, 0.0001)),
            yDown: CGFloat((maxY - vertex.y) / max(maxY - minY, 0.0001))
        )
    }
}

private struct NormalizedFacePoint {
    let x: CGFloat
    let yDown: CGFloat
}

private struct VertexSample {
    let facePoint: NormalizedFacePoint
    let texturePoint: UVPoint
}

private struct UVBounds {
    let minU: CGFloat
    let maxU: CGFloat
    let minV: CGFloat
    let maxV: CGFloat

    nonisolated init(points: [UVPoint]) {
        minU = points.map(\.u).min() ?? 0
        maxU = points.map(\.u).max() ?? 0
        minV = points.map(\.v).min() ?? 0
        maxV = points.map(\.v).max() ?? 0
    }

    nonisolated init(minU: CGFloat, maxU: CGFloat, minV: CGFloat, maxV: CGFloat) {
        self.minU = minU
        self.maxU = maxU
        self.minV = minV
        self.maxV = maxV
    }

    nonisolated var width: CGFloat {
        maxU - minU
    }

    nonisolated var height: CGFloat {
        maxV - minV
    }

    nonisolated var midU: CGFloat {
        (minU + maxU) * 0.5
    }

    nonisolated var midY: CGFloat {
        (minV + maxV) * 0.5
    }

    nonisolated var isValid: Bool {
        width > 0.015 && height > 0.008
    }

    nonisolated func expanded(x: CGFloat, y: CGFloat) -> UVBounds {
        UVBounds(
            minU: minU - width * x,
            maxU: maxU + width * x,
            minV: minV - height * y,
            maxV: maxV + height * y
        )
    }

    nonisolated func clamped() -> UVBounds {
        UVBounds(
            minU: min(max(minU, 0), 1),
            maxU: min(max(maxU, 0), 1),
            minV: min(max(minV, 0), 1),
            maxV: min(max(maxV, 0), 1)
        )
    }

    nonisolated func clamped(to bounds: UVBounds) -> UVBounds {
        UVBounds(
            minU: max(minU, bounds.minU),
            maxU: min(maxU, bounds.maxU),
            minV: max(minV, bounds.minV),
            maxV: min(maxV, bounds.maxV)
        )
    }
}

private enum FaceSide {
    case left
    case right

    var mirror: CGFloat {
        self == .left ? -1 : 1
    }
}

private enum UVMakeupLandmarks {
    static let faceOval = UVPath {
        UVPoint(0.500, 0.145)
        UVPoint(0.635, 0.174)
        UVPoint(0.732, 0.312)
        UVPoint(0.704, 0.545)
        UVPoint(0.626, 0.742)
        UVPoint(0.500, 0.835)
        UVPoint(0.374, 0.742)
        UVPoint(0.296, 0.545)
        UVPoint(0.268, 0.312)
        UVPoint(0.365, 0.174)
    }

    static let upperLip = UVPath {
        UVPoint(0.386, 0.616)
        UVPoint(0.420, 0.599)
        UVPoint(0.456, 0.590)
        UVPoint(0.482, 0.592)
        UVPoint(0.500, 0.580)
        UVPoint(0.518, 0.592)
        UVPoint(0.544, 0.590)
        UVPoint(0.580, 0.599)
        UVPoint(0.614, 0.616)
        UVPoint(0.570, 0.629)
        UVPoint(0.500, 0.626)
        UVPoint(0.430, 0.629)
    }

    static let lowerLip = UVPath {
        UVPoint(0.386, 0.620)
        UVPoint(0.430, 0.642)
        UVPoint(0.470, 0.657)
        UVPoint(0.500, 0.662)
        UVPoint(0.530, 0.657)
        UVPoint(0.570, 0.642)
        UVPoint(0.614, 0.620)
        UVPoint(0.596, 0.676)
        UVPoint(0.556, 0.700)
        UVPoint(0.500, 0.711)
        UVPoint(0.444, 0.700)
        UVPoint(0.404, 0.676)
    }

    static let lipHighlight = UVPath {
        UVPoint(0.446, 0.635)
        UVPoint(0.486, 0.624)
        UVPoint(0.554, 0.628)
        UVPoint(0.590, 0.642)
    }

    static let lipCrease = UVPath {
        UVPoint(0.398, 0.620)
        UVPoint(0.500, 0.626)
        UVPoint(0.602, 0.620)
    }

    static func cheekCenter(_ side: FaceSide) -> UVPoint {
        mirrored(x: 0.382, y: 0.522, side: side)
    }

    static func eyeCenter(_ side: FaceSide) -> UVPoint {
        mirrored(x: 0.386, y: 0.356, side: side)
    }

    static func eyeShadow(_ side: FaceSide) -> UVPath {
        let points = [
            mirrored(x: 0.304, y: 0.350, side: side),
            mirrored(x: 0.350, y: 0.310, side: side),
            mirrored(x: 0.430, y: 0.310, side: side),
            mirrored(x: 0.478, y: 0.356, side: side),
            mirrored(x: 0.432, y: 0.394, side: side),
            mirrored(x: 0.352, y: 0.394, side: side)
        ]
        return UVPath(points: points)
    }

    static func brow(_ side: FaceSide) -> UVPath {
        let points = [
            mirrored(x: 0.300, y: 0.286, side: side),
            mirrored(x: 0.356, y: 0.260, side: side),
            mirrored(x: 0.440, y: 0.268, side: side),
            mirrored(x: 0.484, y: 0.294, side: side),
            mirrored(x: 0.438, y: 0.306, side: side),
            mirrored(x: 0.354, y: 0.298, side: side)
        ]
        return UVPath(points: points)
    }

    static func noseContour(_ side: FaceSide) -> UVPath {
        let points = [
            mirrored(x: 0.462, y: 0.342, side: side),
            mirrored(x: 0.482, y: 0.360, side: side),
            mirrored(x: 0.474, y: 0.548, side: side),
            mirrored(x: 0.442, y: 0.584, side: side),
            mirrored(x: 0.430, y: 0.558, side: side),
            mirrored(x: 0.452, y: 0.430, side: side)
        ]
        return UVPath(points: points)
    }

    static func cheekContour(_ side: FaceSide) -> UVPath {
        let points = [
            mirrored(x: 0.284, y: 0.500, side: side),
            mirrored(x: 0.334, y: 0.548, side: side),
            mirrored(x: 0.458, y: 0.584, side: side),
            mirrored(x: 0.430, y: 0.628, side: side),
            mirrored(x: 0.322, y: 0.600, side: side),
            mirrored(x: 0.262, y: 0.548, side: side)
        ]
        return UVPath(points: points)
    }

    private static func mirrored(x: CGFloat, y: CGFloat, side: FaceSide) -> UVPoint {
        UVPoint(0.5 + ((x - 0.5) * side.mirror), y)
    }
}

private struct UVPath {
    let points: [UVPoint]

    nonisolated init(points: [UVPoint]) {
        self.points = points
    }

    nonisolated init(@UVPointBuilder _ builder: () -> [UVPoint]) {
        points = builder()
    }

    nonisolated static func lipUpper(in bounds: UVBounds) -> UVPath {
        let left = bounds.minU
        let right = bounds.maxU
        let center = bounds.midU
        let top = bounds.minV
        let bottom = bounds.maxV
        let width = bounds.width
        let height = bounds.height

        return UVPath(points: [
            UVPoint(left, bottom - height * 0.25),
            UVPoint(center - width * 0.31, top + height * 0.26),
            UVPoint(center - width * 0.12, top + height * 0.39),
            UVPoint(center, top + height * 0.23),
            UVPoint(center + width * 0.12, top + height * 0.39),
            UVPoint(center + width * 0.31, top + height * 0.26),
            UVPoint(right, bottom - height * 0.25),
            UVPoint(center + width * 0.24, bottom - height * 0.08),
            UVPoint(center, bottom - height * 0.15),
            UVPoint(center - width * 0.24, bottom - height * 0.08)
        ])
    }

    nonisolated static func lipLower(in bounds: UVBounds) -> UVPath {
        let left = bounds.minU
        let right = bounds.maxU
        let center = bounds.midU
        let top = bounds.minV
        let bottom = bounds.maxV
        let width = bounds.width
        let height = bounds.height

        return UVPath(points: [
            UVPoint(left, top + height * 0.08),
            UVPoint(center - width * 0.25, top + height * 0.28),
            UVPoint(center, top + height * 0.34),
            UVPoint(center + width * 0.25, top + height * 0.28),
            UVPoint(right, top + height * 0.08),
            UVPoint(center + width * 0.32, bottom - height * 0.08),
            UVPoint(center, bottom - height * 0.03),
            UVPoint(center - width * 0.32, bottom - height * 0.08)
        ])
    }

    nonisolated static func lipHighlight(in bounds: UVBounds) -> UVPath {
        let center = bounds.midU
        let width = bounds.width
        let y = bounds.minV + bounds.height * 0.42

        return UVPath(points: [
            UVPoint(center - width * 0.22, y),
            UVPoint(center - width * 0.07, y - bounds.height * 0.10),
            UVPoint(center + width * 0.10, y - bounds.height * 0.08),
            UVPoint(center + width * 0.24, y + bounds.height * 0.03)
        ])
    }

    nonisolated static func mouthSafetyClip(in bounds: UVBounds) -> UVPath {
        let left = bounds.minU
        let right = bounds.maxU
        let center = bounds.midU
        let top = bounds.minV
        let bottom = bounds.maxV
        let width = bounds.width
        let height = bounds.height

        return UVPath(points: [
            UVPoint(left + width * 0.10, top + height * 0.42),
            UVPoint(center - width * 0.28, top + height * 0.08),
            UVPoint(center, top),
            UVPoint(center + width * 0.28, top + height * 0.08),
            UVPoint(right - width * 0.10, top + height * 0.42),
            UVPoint(right, bounds.midY),
            UVPoint(right - width * 0.12, bottom - height * 0.16),
            UVPoint(center, bottom),
            UVPoint(left + width * 0.12, bottom - height * 0.16),
            UVPoint(left, bounds.midY)
        ])
    }

    nonisolated static func lipCrease(in bounds: UVBounds) -> UVPath {
        let center = bounds.midU
        let width = bounds.width
        let y = bounds.midY

        return UVPath(points: [
            UVPoint(center - width * 0.42, y - bounds.height * 0.015),
            UVPoint(center, y + bounds.height * 0.030),
            UVPoint(center + width * 0.42, y - bounds.height * 0.015)
        ])
    }

    nonisolated func path(in rect: CGRect) -> UIBezierPath {
        let path = UIBezierPath()
        guard let first = points.first else { return path }

        if points.count > 3 {
            let last = points[points.count - 1].point(in: rect)
            let start = midpoint(last, first.point(in: rect))
            path.move(to: start)

            for index in points.indices {
                let current = points[index].point(in: rect)
                let next = points[(index + 1) % points.count].point(in: rect)
                path.addQuadCurve(
                    to: midpoint(current, next),
                    controlPoint: current
                )
            }
            path.close()
        } else {
            path.move(to: first.point(in: rect))
            for point in points.dropFirst() {
                path.addLine(to: point.point(in: rect))
            }
        }

        return path
    }

    nonisolated func scaled(x: CGFloat, y: CGFloat) -> UVPath {
        guard !points.isEmpty else { return self }

        let centerU = points.reduce(CGFloat.zero) { $0 + $1.u } / CGFloat(points.count)
        let centerV = points.reduce(CGFloat.zero) { $0 + $1.v } / CGFloat(points.count)
        let center = UVPoint(centerU, centerV)
        return UVPath(
            points: points.map { point in
                UVPoint(
                    center.u + ((point.u - center.u) * x),
                    center.v + ((point.v - center.v) * y)
                )
            }
        )
    }

    nonisolated private func midpoint(_ a: CGPoint, _ b: CGPoint) -> CGPoint {
        CGPoint(x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5)
    }
}

private struct UVPoint {
    let u: CGFloat
    let v: CGFloat

    nonisolated init(_ u: CGFloat, _ v: CGFloat) {
        self.u = u
        self.v = v
    }

    nonisolated init(_ textureCoordinate: simd_float2) {
        u = CGFloat(textureCoordinate.x)
        v = CGFloat(textureCoordinate.y)
    }

    nonisolated func point(in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + rect.width * u, y: rect.minY + rect.height * v)
    }

    nonisolated func interpolated(to point: UVPoint, amount: CGFloat) -> UVPoint {
        UVPoint(
            u + ((point.u - u) * amount),
            v + ((point.v - v) * amount)
        )
    }
}

@resultBuilder
private enum UVPointBuilder {
    static func buildBlock(_ points: UVPoint...) -> [UVPoint] {
        points
    }
}

private extension UIColor {
    var alpha: CGFloat {
        cgColor.alpha
    }
}
