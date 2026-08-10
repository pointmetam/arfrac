//
//  ARKitLipParsingOverlay.swift
//  arface
//
//  Created by lee on 2026/5/25.
//

import ARKit
import CoreML
import UIKit

struct ARKitLipParsingRequest: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let roi: CGRect
    let topLeft: CGPoint
    let topRight: CGPoint
    let bottomLeft: CGPoint
    let bottomRight: CGPoint
}

struct ARKitLipMask: @unchecked Sendable {
    let frame: CGRect
    let width: Int
    let height: Int
    let alpha: [UInt8]

    func displayMask(outputScale: Int = 3) -> UVLipAlphaMask {
        guard width > 0, height > 0, alpha.count == width * height else {
            return UVLipAlphaMask(width: width, height: height, alpha: alpha)
        }

        let baseMask = UVLipAlphaMask(width: width, height: height, alpha: alpha)
        let creasePreserved = Self.preserveClosedMouthCrease(in: baseMask)
        return creasePreserved.productContourSmoothedForMakeup(
            edgeThreshold: 32,
            smoothingIterations: 2,
            edgeFeatherPixels: 0,
            outputScale: max(outputScale, 1)
        )
    }

    func alphaMaskImage() -> CGImage? {
        let displayMask = displayMask()
        guard
            displayMask.width > 0,
            displayMask.height > 0,
            displayMask.alpha.count == displayMask.width * displayMask.height
        else {
            return nil
        }

        var mask = [UInt8](repeating: 0, count: displayMask.width * displayMask.height)
        for index in displayMask.alpha.indices {
            let rawMaskAlpha = CGFloat(displayMask.alpha[index]) / 255
            let maskAlpha = Self.productFeatheredAlpha(rawMaskAlpha)
            mask[index] = UInt8(min(max(maskAlpha * 255, 0), 255))
        }

        let data = Data(mask)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        return CGImage(
            maskWidth: displayMask.width,
            height: displayMask.height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: displayMask.width,
            provider: provider,
            decode: nil,
            shouldInterpolate: true
        )
    }

    func tintedImage(color: UIColor) -> CGImage? {
        let displayMask = displayMask()
        guard
            displayMask.width > 0,
            displayMask.height > 0,
            displayMask.alpha.count == displayMask.width * displayMask.height
        else {
            return nil
        }

        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var colorAlpha: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &colorAlpha)

        var pixels = [UInt8](repeating: 0, count: displayMask.width * displayMask.height * 4)
        for index in displayMask.alpha.indices {
            let rawMaskAlpha = CGFloat(displayMask.alpha[index]) / 255
            let maskAlpha = Self.productFeatheredAlpha(rawMaskAlpha)
            let lipstickAlpha = min(max(colorAlpha * 1.08, 0), 1)
            let combinedAlpha = min(max(maskAlpha * lipstickAlpha, 0), 1)
            let pixelOffset = index * 4
            pixels[pixelOffset] = UInt8(min(max(red * combinedAlpha * 255, 0), 255))
            pixels[pixelOffset + 1] = UInt8(min(max(green * combinedAlpha * 255, 0), 255))
            pixels[pixelOffset + 2] = UInt8(min(max(blue * combinedAlpha * 255, 0), 255))
            pixels[pixelOffset + 3] = UInt8(min(max(combinedAlpha * 255, 0), 255))
        }

        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        return CGImage(
            width: displayMask.width,
            height: displayMask.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: displayMask.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    private static func productFeatheredAlpha(_ alpha: CGFloat) -> CGFloat {
        let low: CGFloat = 0.08
        let high: CGFloat = 0.84
        let normalized = min(max((alpha - low) / (high - low), 0), 1)
        let smoothed = normalized * normalized * (3 - 2 * normalized)
        return min(max(smoothed, 0), 1)
    }

    private static func preserveClosedMouthCrease(in mask: UVLipAlphaMask, threshold: UInt8 = 28) -> UVLipAlphaMask {
        guard mask.width > 8, mask.height > 8, mask.alpha.count == mask.width * mask.height else { return mask }

        var topByColumn = [Int](repeating: Int.max, count: mask.width)
        var bottomByColumn = [Int](repeating: -1, count: mask.width)
        var minY = Int.max
        var maxY = -1
        for y in 0..<mask.height {
            let rowOffset = y * mask.width
            for x in 0..<mask.width where mask.alpha[rowOffset + x] > threshold {
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
            return mask
        }

        let lipHeight = maxY - minY + 1
        let lipWidth = maxX - minX + 1
        let centerX = CGFloat(minX + maxX) * 0.5
        let halfWidth = max(CGFloat(lipWidth) * 0.5, 1)
        let compactLip = CGFloat(lipHeight) / CGFloat(max(lipWidth, 1)) < 0.58

        var gapColumns = 0
        for x in minX...maxX {
            let top = topByColumn[x]
            let bottom = bottomByColumn[x]
            guard bottom - top >= 5 else { continue }

            let searchTop = top + max(1, (bottom - top) / 4)
            let searchBottom = bottom - max(1, (bottom - top) / 4)
            guard searchBottom > searchTop else { continue }

            let hasGap = (searchTop...searchBottom).contains { y in
                mask.alpha[y * mask.width + x] <= threshold
            }
            if hasGap {
                gapColumns += 1
            }
        }
        if gapColumns >= max(6, lipWidth / 5) {
            return mask
        }

        var output = mask.alpha
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
                let value = Int(mask.alpha[y * mask.width + x])
                if value < valleyAlpha {
                    valleyAlpha = value
                    valleyY = y
                }
            }

            let upperShoulderY = max(top, valleyY - 1)
            let lowerShoulderY = min(bottom, valleyY + 1)
            let shoulderAlpha = max(
                Int(mask.alpha[upperShoulderY * mask.width + x]),
                Int(mask.alpha[lowerShoulderY * mask.width + x])
            )
            let valleyDepth = shoulderAlpha - valleyAlpha
            let centerProgress = 1 - min(abs(CGFloat(x) - centerX) / halfWidth, 1)
            let centerWeight = smoothstep(centerProgress)
            let shouldCut = valleyDepth >= 16 || (compactLip && centerWeight > 0.22 && valleyAlpha < 196)
            guard shouldCut else { continue }

            let cutStrength = min(max(0.22 + (centerWeight * 0.78), 0), 1)
            let centerIndex = valleyY * mask.width + x
            output[centerIndex] = UInt8(min(max(CGFloat(output[centerIndex]) * (1 - cutStrength), 0), 255))

            if centerWeight > 0.58, valleyY > top {
                let upperIndex = (valleyY - 1) * mask.width + x
                output[upperIndex] = UInt8(min(max(CGFloat(output[upperIndex]) * (1 - cutStrength * 0.68), 0), 255))
            }
            if centerWeight > 0.58, valleyY < bottom {
                let lowerIndex = (valleyY + 1) * mask.width + x
                output[lowerIndex] = UInt8(min(max(CGFloat(output[lowerIndex]) * (1 - cutStrength * 0.68), 0), 255))
            }

            if centerWeight > 0.14 {
                cutColumns += 1
            }
        }

        guard cutColumns >= max(5, lipWidth / 6) else { return mask }
        return UVLipAlphaMask(width: mask.width, height: mask.height, alpha: output)
    }

    private static func smoothstep(_ value: CGFloat) -> CGFloat {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }
}

struct ARKitFaceParsingMask: @unchecked Sendable {
    static let debugPalette: [SIMD4<UInt8>] = [
        SIMD4<UInt8>(0, 0, 0, 0),
        SIMD4<UInt8>(255, 178, 132, 78),
        SIMD4<UInt8>(136, 78, 255, 172),
        SIMD4<UInt8>(67, 214, 255, 156),
        SIMD4<UInt8>(86, 226, 145, 128),
        SIMD4<UInt8>(255, 42, 82, 214),
        SIMD4<UInt8>(255, 92, 138, 214),
        SIMD4<UInt8>(72, 102, 255, 186),
        SIMD4<UInt8>(42, 44, 52, 116)
    ]

    let frame: CGRect
    let width: Int
    let height: Int
    let classIds: [UInt8]
    let confidence: [UInt8]

    func lipstickMask() -> ARKitLipMask? {
        guard width > 0, height > 0, classIds.count == width * height, confidence.count == width * height else {
            return nil
        }

        var alpha = [UInt8](repeating: 0, count: width * height)
        var activePixelCount = 0

        for index in classIds.indices {
            let classId = classIds[index]
            let pixelConfidence = confidence[index]

            if classId == 5 || classId == 6 {
                guard pixelConfidence >= 8 else { continue }

                let normalized = CGFloat(pixelConfidence) / 255
                let opacity = min(max((normalized - 0.04) / 0.50, 0), 1)
                guard opacity > 0.04 else { continue }

                alpha[index] = UInt8(min(max(opacity * 245, 0), 245))
                activePixelCount += 1
            } else if classId == 7, pixelConfidence > 8 {
                alpha[index] = 0
            }
        }

        let coverage = CGFloat(activePixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.002, coverage < 0.18 else { return nil }

        alpha = Self.removeTinyComponents(alpha: alpha, width: width, height: height, threshold: 16)
        alpha = Self.softlyExpanded(alpha: alpha, width: width, height: height, radius: 1)
        alpha = Self.blurred(alpha: alpha, width: width, height: height, radius: 1)

        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    func debugImage() -> CGImage? {
        guard width > 0, height > 0, classIds.count == width * height, confidence.count == width * height else {
            return nil
        }

        let pixels = Self.rgbaPixels(classIds: classIds, confidence: confidence)
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

    private static func rgbaPixels(classIds: [UInt8], confidence: [UInt8]) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: classIds.count * 4)
        for index in classIds.indices {
            let classId = Int(classIds[index])
            guard classId > 0, classId < debugPalette.count else { continue }

            let conf = UInt16(confidence[index])
            guard conf > 12 else { continue }

            let color = debugPalette[classId]
            let alpha = UInt16(color.w) * conf / 255
            let offset = index * 4
            pixels[offset] = UInt8(UInt16(color.x) * alpha / 255)
            pixels[offset + 1] = UInt8(UInt16(color.y) * alpha / 255)
            pixels[offset + 2] = UInt8(UInt16(color.z) * alpha / 255)
            pixels[offset + 3] = UInt8(alpha)
        }

        return pixels
    }

    private static func removeTinyComponents(alpha: [UInt8], width: Int, height: Int, threshold: UInt8) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else { return alpha }

        let minimumPixels = max(16, Int(CGFloat(width * height) * 0.00028))
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

                if x > 0 {
                    let neighbor = y * width + x - 1
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
                    }
                }
                if x < width - 1 {
                    let neighbor = y * width + x + 1
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
                    }
                }
                if y > 0 {
                    let neighbor = (y - 1) * width + x
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
                    }
                }
                if y < height - 1 {
                    let neighbor = (y + 1) * width + x
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
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

                if neighborMax > 48 {
                    output[y * width + x] = UInt8(min(max(CGFloat(neighborMax) * 0.42, 0), 128))
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
}

enum FaceParsingClass: UInt8, CaseIterable, Hashable {
    case background = 0
    case skin = 1
    case eyebrow = 2
    case eye = 3
    case nose = 4
    case upperLip = 5
    case lowerLip = 6
    case innerMouth = 7
    case hair = 8
}

struct UVFaceParsingMask: @unchecked Sendable {
    let width: Int
    let height: Int
    let classIds: [UInt8]
    let confidence: [UInt8]

    func temporallySmoothed(previous: UVFaceParsingMask?, response: CGFloat = 0.42) -> UVFaceParsingMask {
        guard
            let previous,
            previous.width == width,
            previous.height == height,
            previous.classIds.count == classIds.count,
            previous.confidence.count == confidence.count
        else {
            return self
        }

        let newWeight = min(max(response, 0.05), 1)
        let oldWeight = 1 - newWeight
        var smoothedClassIds = classIds
        var smoothedConfidence = confidence

        for index in classIds.indices {
            let newClass = classIds[index]
            let oldClass = previous.classIds[index]
            let newConfidence = CGFloat(confidence[index])
            let oldConfidence = CGFloat(previous.confidence[index])

            if newClass == oldClass {
                smoothedClassIds[index] = newClass
                smoothedConfidence[index] = UInt8(min(max((oldConfidence * oldWeight) + (newConfidence * newWeight), 0), 255))
                continue
            }

            if newConfidence > oldConfidence + 28 {
                smoothedClassIds[index] = newClass
                smoothedConfidence[index] = UInt8(min(max((oldConfidence * 0.30) + (newConfidence * 0.70), 0), 255))
            } else {
                smoothedClassIds[index] = oldClass
                smoothedConfidence[index] = UInt8(min(max(oldConfidence * 0.88, newConfidence * 0.34), 255))
            }
        }

        return UVFaceParsingMask(
            width: width,
            height: height,
            classIds: smoothedClassIds,
            confidence: smoothedConfidence
        )
    }

    func alphaMask(
        for includedClasses: Set<FaceParsingClass>,
        minimumConfidence: UInt8 = 26,
        minimumCoverage: CGFloat = 0.001,
        maximumCoverage: CGFloat = 0.24,
        expansionRadius: Int = 1,
        blurRadius: Int = 1
    ) -> UVLipAlphaMask? {
        guard
            width > 0,
            height > 0,
            classIds.count == width * height,
            confidence.count == width * height,
            !includedClasses.isEmpty
        else {
            return nil
        }

        let includedIds = Set(includedClasses.map(\.rawValue))
        var alpha = [UInt8](repeating: 0, count: width * height)
        var activePixelCount = 0

        for index in classIds.indices where includedIds.contains(classIds[index]) {
            let pixelConfidence = confidence[index]
            guard pixelConfidence >= minimumConfidence else { continue }

            let normalized = CGFloat(pixelConfidence) / 255
            let opacity = min(max((normalized - 0.08) / 0.52, 0), 1)
            guard opacity > 0.04 else { continue }

            alpha[index] = UInt8(min(max(opacity * 255, 0), 255))
            activePixelCount += 1
        }

        let coverage = CGFloat(activePixelCount) / CGFloat(max(width * height, 1))
        guard coverage >= minimumCoverage, coverage <= maximumCoverage else { return nil }

        alpha = softlyExpanded(alpha: alpha, width: width, height: height, radius: expansionRadius)
        alpha = blurred(alpha: alpha, width: width, height: height, radius: blurRadius)
        return UVLipAlphaMask(width: width, height: height, alpha: alpha)
    }

    private func softlyExpanded(alpha: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        var output = alpha
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] == 0 {
                var neighborMax = 0
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
                    for sampleX in max(0, x - radius)...min(width - 1, x + radius) {
                        neighborMax = max(neighborMax, Int(alpha[sampleY * width + sampleX]))
                    }
                }

                if neighborMax > 54 {
                    output[y * width + x] = UInt8(min(max(CGFloat(neighborMax) * 0.46, 0), 135))
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

struct ARKitFaceUVProjectionMap: @unchecked Sendable {
    private static let textureSize = 256

    struct Triangle: @unchecked Sendable {
        let uv0: CGPoint
        let uv1: CGPoint
        let uv2: CGPoint
        let screen0: CGPoint
        let screen1: CGPoint
        let screen2: CGPoint
    }

    let triangles: [Triangle]

    func uvMask(from screenMask: ARKitFaceParsingMask) -> UVFaceParsingMask? {
        guard !triangles.isEmpty else { return nil }

        let size = Self.textureSize
        var classIds = [UInt8](repeating: 0, count: size * size)
        var confidence = [UInt8](repeating: 0, count: size * size)

        for triangle in triangles {
            rasterize(triangle: triangle, screenMask: screenMask, classIds: &classIds, confidence: &confidence, size: size)
        }

        let activePixels = confidence.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
        let coverage = CGFloat(activePixels) / CGFloat(max(size * size, 1))
        guard coverage > 0.035, coverage < 0.74 else { return nil }

        return UVFaceParsingMask(width: size, height: size, classIds: classIds, confidence: confidence)
    }

    func uvLipAlphaMask(from screenMask: ARKitLipMask) -> UVLipAlphaMask? {
        guard
            !triangles.isEmpty,
            screenMask.width > 1,
            screenMask.height > 1,
            screenMask.alpha.count == screenMask.width * screenMask.height
        else {
            return nil
        }

        let size = Self.textureSize
        var alpha = [UInt8](repeating: 0, count: size * size)
        // Curve masks are later gated again against the base UV-4C mask, so this UV crop can stay permissive.
        let lipUVBounds = CGRect(x: 0.32, y: 0.50, width: 0.36, height: 0.28)

        for triangle in triangles {
            rasterizeLipAlpha(
                triangle: triangle,
                screenMask: screenMask,
                alpha: &alpha,
                size: size,
                lipUVBounds: lipUVBounds
            )
        }

        let activePixels = alpha.reduce(0) { $0 + ($1 > 8 ? 1 : 0) }
        let coverage = CGFloat(activePixels) / CGFloat(max(size * size, 1))
        // Keep the projection gate loose; the downstream curve/base overlap gate is the real safety check.
        guard activePixels > 24, coverage > 0.00035, coverage < 0.20 else { return nil }

        return UVLipAlphaMask(width: size, height: size, alpha: alpha)
    }

    func uvLipAlphaMask(fromScreenCurves curves: [[CGPoint]]) -> UVLipAlphaMask? {
        guard curves.count == 4, !triangles.isEmpty else { return nil }

        let size = Self.textureSize
        let lipUVBounds = CGRect(x: 0.32, y: 0.50, width: 0.36, height: 0.28)
        let samples = makeLipScreenUVSamples(lipUVBounds: lipUVBounds, size: size)
        guard samples.count > 200 else { return nil }
        let expandedLipUVBounds = lipUVBounds.insetBy(dx: -0.025, dy: -0.025)
        let mappedCurves = curves.map { curve in
            mapScreenCurveToUV(curve, lipUVBounds: expandedLipUVBounds, fallbackSamples: samples)
        }
        guard mappedCurves.allSatisfy({ $0.count >= 8 }) else { return nil }
        let stabilizedCurves = stabilizedMappedLipCurves(mappedCurves, lipUVBounds: expandedLipUVBounds)

        var alpha = [UInt8](repeating: 0, count: size * size)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &alpha,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        context.setShouldAntialias(true)
        context.setFillColor(gray: 1, alpha: 1)

        let maxCoord = CGFloat(max(size - 1, 1))
        let toTexturePoint: (CGPoint) -> CGPoint = { uv in
            CGPoint(
                x: min(max(uv.x, 0), 1) * maxCoord,
                y: min(max(uv.y, 0), 1) * maxCoord
            )
        }

        var upperLip = stabilizedCurves[0].map(toTexturePoint)
        upperLip.append(contentsOf: stabilizedCurves[1].reversed().map(toTexturePoint))
        drawUVPolygon(points: upperLip, in: context)
        context.fillPath()

        var lowerLip = stabilizedCurves[3].map(toTexturePoint)
        lowerLip.append(contentsOf: stabilizedCurves[2].reversed().map(toTexturePoint))
        drawUVPolygon(points: lowerLip, in: context)
        context.fillPath()

        let activePixels = alpha.reduce(0) { $0 + ($1 > 8 ? 1 : 0) }
        let coverage = CGFloat(activePixels) / CGFloat(max(size * size, 1))
        guard activePixels > 24, coverage > 0.00035, coverage < 0.20 else { return nil }

        return UVLipAlphaMask(width: size, height: size, alpha: alpha)
    }

    private func rasterize(
        triangle: Triangle,
        screenMask: ARKitFaceParsingMask,
        classIds: inout [UInt8],
        confidence: inout [UInt8],
        size: Int
    ) {
        let p0 = CGPoint(x: triangle.uv0.x * CGFloat(size), y: triangle.uv0.y * CGFloat(size))
        let p1 = CGPoint(x: triangle.uv1.x * CGFloat(size), y: triangle.uv1.y * CGFloat(size))
        let p2 = CGPoint(x: triangle.uv2.x * CGFloat(size), y: triangle.uv2.y * CGFloat(size))

        let minX = min(max(Int(floor(min(p0.x, p1.x, p2.x))), 0), size - 1)
        let maxX = min(max(Int(ceil(max(p0.x, p1.x, p2.x))), 0), size - 1)
        let minY = min(max(Int(floor(min(p0.y, p1.y, p2.y))), 0), size - 1)
        let maxY = min(max(Int(ceil(max(p0.y, p1.y, p2.y))), 0), size - 1)
        guard minX <= maxX, minY <= maxY else { return }

        let denominator = ((p1.y - p2.y) * (p0.x - p2.x)) + ((p2.x - p1.x) * (p0.y - p2.y))
        guard abs(denominator) > 0.0001 else { return }

        for y in minY...maxY {
            for x in minX...maxX {
                let point = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
                let w0 = (((p1.y - p2.y) * (point.x - p2.x)) + ((p2.x - p1.x) * (point.y - p2.y))) / denominator
                let w1 = (((p2.y - p0.y) * (point.x - p2.x)) + ((p0.x - p2.x) * (point.y - p2.y))) / denominator
                let w2 = 1 - w0 - w1

                guard w0 >= -0.001, w1 >= -0.001, w2 >= -0.001 else { continue }

                let screenPoint = CGPoint(
                    x: triangle.screen0.x * w0 + triangle.screen1.x * w1 + triangle.screen2.x * w2,
                    y: triangle.screen0.y * w0 + triangle.screen1.y * w1 + triangle.screen2.y * w2
                )
                let sample = screenMask.sample(at: screenPoint)
                guard sample.confidence > 0, sample.classId > 0 else { continue }

                let index = y * size + x
                if sample.confidence >= confidence[index] {
                    classIds[index] = sample.classId
                    confidence[index] = sample.confidence
                }
            }
        }
    }

    private func rasterizeLipAlpha(
        triangle: Triangle,
        screenMask: ARKitLipMask,
        alpha: inout [UInt8],
        size: Int,
        lipUVBounds: CGRect
    ) {
        let p0 = CGPoint(x: triangle.uv0.x * CGFloat(size), y: triangle.uv0.y * CGFloat(size))
        let p1 = CGPoint(x: triangle.uv1.x * CGFloat(size), y: triangle.uv1.y * CGFloat(size))
        let p2 = CGPoint(x: triangle.uv2.x * CGFloat(size), y: triangle.uv2.y * CGFloat(size))

        let minX = min(max(Int(floor(min(p0.x, p1.x, p2.x))), 0), size - 1)
        let maxX = min(max(Int(ceil(max(p0.x, p1.x, p2.x))), 0), size - 1)
        let minY = min(max(Int(floor(min(p0.y, p1.y, p2.y))), 0), size - 1)
        let maxY = min(max(Int(ceil(max(p0.y, p1.y, p2.y))), 0), size - 1)
        guard minX <= maxX, minY <= maxY else { return }

        let denominator = ((p1.y - p2.y) * (p0.x - p2.x)) + ((p2.x - p1.x) * (p0.y - p2.y))
        guard abs(denominator) > 0.0001 else { return }

        for y in minY...maxY {
            for x in minX...maxX {
                let uvPoint = CGPoint(
                    x: (CGFloat(x) + 0.5) / CGFloat(size),
                    y: (CGFloat(y) + 0.5) / CGFloat(size)
                )
                guard lipUVBounds.contains(uvPoint) else { continue }

                let point = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
                let w0 = (((p1.y - p2.y) * (point.x - p2.x)) + ((p2.x - p1.x) * (point.y - p2.y))) / denominator
                let w1 = (((p2.y - p0.y) * (point.x - p2.x)) + ((p0.x - p2.x) * (point.y - p2.y))) / denominator
                let w2 = 1 - w0 - w1

                guard w0 >= -0.001, w1 >= -0.001, w2 >= -0.001 else { continue }

                let screenPoint = CGPoint(
                    x: triangle.screen0.x * w0 + triangle.screen1.x * w1 + triangle.screen2.x * w2,
                    y: triangle.screen0.y * w0 + triangle.screen1.y * w1 + triangle.screen2.y * w2
                )
                let sample = screenMask.alphaValue(at: screenPoint)
                guard sample > 2 else { continue }

                let index = y * size + x
                if sample > alpha[index] {
                    alpha[index] = sample
                }
            }
        }
    }

    private func makeLipScreenUVSamples(lipUVBounds: CGRect, size: Int) -> [(screen: CGPoint, uv: CGPoint)] {
        var samples: [(screen: CGPoint, uv: CGPoint)] = []
        samples.reserveCapacity(6000)
        for triangle in triangles {
            let p0 = CGPoint(x: triangle.uv0.x * CGFloat(size), y: triangle.uv0.y * CGFloat(size))
            let p1 = CGPoint(x: triangle.uv1.x * CGFloat(size), y: triangle.uv1.y * CGFloat(size))
            let p2 = CGPoint(x: triangle.uv2.x * CGFloat(size), y: triangle.uv2.y * CGFloat(size))
            let minX = min(max(Int(floor(min(p0.x, p1.x, p2.x))), 0), size - 1)
            let maxX = min(max(Int(ceil(max(p0.x, p1.x, p2.x))), 0), size - 1)
            let minY = min(max(Int(floor(min(p0.y, p1.y, p2.y))), 0), size - 1)
            let maxY = min(max(Int(ceil(max(p0.y, p1.y, p2.y))), 0), size - 1)
            guard minX <= maxX, minY <= maxY else { continue }

            let denominator = ((p1.y - p2.y) * (p0.x - p2.x)) + ((p2.x - p1.x) * (p0.y - p2.y))
            guard abs(denominator) > 0.0001 else { continue }

            for y in stride(from: minY, through: maxY, by: 2) {
                for x in stride(from: minX, through: maxX, by: 2) {
                    let uvPoint = CGPoint(
                        x: (CGFloat(x) + 0.5) / CGFloat(size),
                        y: (CGFloat(y) + 0.5) / CGFloat(size)
                    )
                    guard lipUVBounds.contains(uvPoint) else { continue }

                    let point = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
                    let w0 = (((p1.y - p2.y) * (point.x - p2.x)) + ((p2.x - p1.x) * (point.y - p2.y))) / denominator
                    let w1 = (((p2.y - p0.y) * (point.x - p2.x)) + ((p0.x - p2.x) * (point.y - p2.y))) / denominator
                    let w2 = 1 - w0 - w1
                    guard w0 >= -0.001, w1 >= -0.001, w2 >= -0.001 else { continue }

                    let screenPoint = CGPoint(
                        x: triangle.screen0.x * w0 + triangle.screen1.x * w1 + triangle.screen2.x * w2,
                        y: triangle.screen0.y * w0 + triangle.screen1.y * w1 + triangle.screen2.y * w2
                    )
                    samples.append((screen: screenPoint, uv: uvPoint))
                }
            }
        }
        return samples
    }

    private func mapScreenCurveToUV(
        _ curve: [CGPoint],
        lipUVBounds: CGRect,
        fallbackSamples: [(screen: CGPoint, uv: CGPoint)]
    ) -> [CGPoint] {
        guard !curve.isEmpty else { return [] }

        var mapped = [CGPoint?](repeating: nil, count: curve.count)
        for (index, point) in curve.enumerated() {
            mapped[index] = uvPoint(forScreenPoint: point, lipUVBounds: lipUVBounds)
                ?? uvPoint(nearestToScreenPoint: point, samples: fallbackSamples)
        }

        for index in mapped.indices where mapped[index] == nil {
            let previous = stride(from: index - 1, through: 0, by: -1).first(where: { mapped[$0] != nil })
            let next = stride(from: index + 1, to: mapped.count, by: 1).first(where: { mapped[$0] != nil })
            switch (previous, next) {
            case let (previous?, next?):
                let progress = CGFloat(index - previous) / CGFloat(max(next - previous, 1))
                let start = mapped[previous]!
                let end = mapped[next]!
                mapped[index] = CGPoint(
                    x: start.x + (end.x - start.x) * progress,
                    y: start.y + (end.y - start.y) * progress
                )
            case let (previous?, nil):
                mapped[index] = mapped[previous]
            case let (nil, next?):
                mapped[index] = mapped[next]
            case (nil, nil):
                break
            }
        }

        return mapped.compactMap { point in
            guard let point else { return nil }
            return clamp(point: point, to: lipUVBounds)
        }
    }

    private func uvPoint(forScreenPoint point: CGPoint, lipUVBounds: CGRect) -> CGPoint? {
        let expandedBounds = lipUVBounds.insetBy(dx: -0.01, dy: -0.01)
        var bestMatch: (score: CGFloat, uv: CGPoint)?
        for triangle in triangles {
            let uvBounds = CGRect(
                x: min(triangle.uv0.x, triangle.uv1.x, triangle.uv2.x),
                y: min(triangle.uv0.y, triangle.uv1.y, triangle.uv2.y),
                width: max(triangle.uv0.x, triangle.uv1.x, triangle.uv2.x) - min(triangle.uv0.x, triangle.uv1.x, triangle.uv2.x),
                height: max(triangle.uv0.y, triangle.uv1.y, triangle.uv2.y) - min(triangle.uv0.y, triangle.uv1.y, triangle.uv2.y)
            )
            guard uvBounds.intersects(expandedBounds) else { continue }
            guard let barycentric = barycentricWeights(point: point, p0: triangle.screen0, p1: triangle.screen1, p2: triangle.screen2) else {
                continue
            }
            let epsilon: CGFloat = 0.004
            guard barycentric.w0 >= -epsilon, barycentric.w1 >= -epsilon, barycentric.w2 >= -epsilon else { continue }

            let uvPoint = CGPoint(
                x: triangle.uv0.x * barycentric.w0 + triangle.uv1.x * barycentric.w1 + triangle.uv2.x * barycentric.w2,
                y: triangle.uv0.y * barycentric.w0 + triangle.uv1.y * barycentric.w1 + triangle.uv2.y * barycentric.w2
            )
            guard expandedBounds.contains(uvPoint) else { continue }

            let edgeMargin = min(barycentric.w0, barycentric.w1, barycentric.w2)
            let score = edgeMargin
            if let bestMatch, bestMatch.score >= score {
                continue
            }
            bestMatch = (score, uvPoint)
        }

        return bestMatch?.uv
    }

    private func uvPoint(nearestToScreenPoint point: CGPoint, samples: [(screen: CGPoint, uv: CGPoint)]) -> CGPoint? {
        var best = [(distance: CGFloat, uv: CGPoint)]()
        best.reserveCapacity(3)

        for sample in samples {
            let dx = sample.screen.x - point.x
            let dy = sample.screen.y - point.y
            let distance = dx * dx + dy * dy

            if best.count < 3 {
                best.append((distance, sample.uv))
                best.sort { $0.distance < $1.distance }
            } else if let last = best.last, distance < last.distance {
                best[best.count - 1] = (distance, sample.uv)
                best.sort { $0.distance < $1.distance }
            }
        }

        guard let nearest = best.first, nearest.distance < 28 * 28 else { return nil }
        let totalWeight = best.reduce(CGFloat.zero) { partial, item in
            partial + (1 / max(item.distance, 1))
        }
        guard totalWeight > 0 else { return nearest.uv }

        let uvX = best.reduce(CGFloat.zero) { partial, item in
            partial + item.uv.x * (1 / max(item.distance, 1))
        } / totalWeight
        let uvY = best.reduce(CGFloat.zero) { partial, item in
            partial + item.uv.y * (1 / max(item.distance, 1))
        } / totalWeight
        return CGPoint(x: uvX, y: uvY)
    }

    private func stabilizedMappedLipCurves(_ curves: [[CGPoint]], lipUVBounds: CGRect) -> [[CGPoint]] {
        guard curves.count == 4 else { return curves }

        var stabilized = curves.map { smoothedMappedCurve($0, lipUVBounds: lipUVBounds) }
        let sharedCount = stabilized.map(\.count).min() ?? 0
        guard sharedCount > 0 else { return stabilized }

        let contourGap = lipUVBounds.height * 0.010
        let mouthGap = lipUVBounds.height * 0.016
        for index in 0..<sharedCount {
            var upperOuter = stabilized[0][index]
            var upperInner = stabilized[1][index]
            var lowerInner = stabilized[2][index]
            var lowerOuter = stabilized[3][index]

            if upperOuter.y > upperInner.y - contourGap {
                let center = (upperOuter.y + upperInner.y) * 0.5
                upperOuter.y = center - contourGap * 0.5
                upperInner.y = center + contourGap * 0.5
            }
            if lowerInner.y > lowerOuter.y - contourGap {
                let center = (lowerInner.y + lowerOuter.y) * 0.5
                lowerInner.y = center - contourGap * 0.5
                lowerOuter.y = center + contourGap * 0.5
            }
            if upperInner.y > lowerInner.y - mouthGap {
                let center = (upperInner.y + lowerInner.y) * 0.5
                upperInner.y = center - mouthGap * 0.5
                lowerInner.y = center + mouthGap * 0.5
            }

            stabilized[0][index] = clamp(point: upperOuter, to: lipUVBounds)
            stabilized[1][index] = clamp(point: upperInner, to: lipUVBounds)
            stabilized[2][index] = clamp(point: lowerInner, to: lipUVBounds)
            stabilized[3][index] = clamp(point: lowerOuter, to: lipUVBounds)
        }

        return stabilized.map { enforceMonotonicX($0, lipUVBounds: lipUVBounds) }
    }

    private func smoothedMappedCurve(_ curve: [CGPoint], lipUVBounds: CGRect) -> [CGPoint] {
        guard curve.count > 2 else {
            return curve.map { clamp(point: $0, to: lipUVBounds) }
        }

        var smoothed = curve.map { clamp(point: $0, to: lipUVBounds) }
        for _ in 0..<2 {
            var output = smoothed
            for index in 1..<(smoothed.count - 1) {
                let previous = smoothed[index - 1]
                let current = smoothed[index]
                let next = smoothed[index + 1]
                output[index] = clamp(
                    point: CGPoint(
                        x: previous.x * 0.22 + current.x * 0.56 + next.x * 0.22,
                        y: previous.y * 0.22 + current.y * 0.56 + next.y * 0.22
                    ),
                    to: lipUVBounds
                )
            }
            smoothed = output
        }
        return enforceMonotonicX(smoothed, lipUVBounds: lipUVBounds)
    }

    private func enforceMonotonicX(_ curve: [CGPoint], lipUVBounds: CGRect) -> [CGPoint] {
        guard curve.count > 1 else { return curve }

        var output = curve
        let slack = lipUVBounds.width * 0.0025
        output[0] = clamp(point: output[0], to: lipUVBounds)
        for index in 1..<output.count {
            var point = clamp(point: output[index], to: lipUVBounds)
            point.x = max(point.x, output[index - 1].x - slack)
            output[index] = clamp(point: point, to: lipUVBounds)
        }
        return output
    }

    private func barycentricWeights(
        point: CGPoint,
        p0: CGPoint,
        p1: CGPoint,
        p2: CGPoint
    ) -> (w0: CGFloat, w1: CGFloat, w2: CGFloat)? {
        let denominator = ((p1.y - p2.y) * (p0.x - p2.x)) + ((p2.x - p1.x) * (p0.y - p2.y))
        guard abs(denominator) > 0.0001 else { return nil }

        let w0 = (((p1.y - p2.y) * (point.x - p2.x)) + ((p2.x - p1.x) * (point.y - p2.y))) / denominator
        let w1 = (((p2.y - p0.y) * (point.x - p2.x)) + ((p0.x - p2.x) * (point.y - p2.y))) / denominator
        let w2 = 1 - w0 - w1
        return (w0, w1, w2)
    }

    private func clamp(point: CGPoint, to bounds: CGRect) -> CGPoint {
        CGPoint(
            x: min(max(point.x, bounds.minX), bounds.maxX),
            y: min(max(point.y, bounds.minY), bounds.maxY)
        )
    }
}

private func drawUVPolygon(points: [CGPoint], in context: CGContext) {
    guard let first = points.first else { return }
    context.move(to: first)
    for point in points.dropFirst() {
        context.addLine(to: point)
    }
    context.closePath()
}

private extension ARKitFaceParsingMask {
    func sample(at point: CGPoint) -> (classId: UInt8, confidence: UInt8) {
        guard
            frame.width > 1,
            frame.height > 1,
            frame.contains(point),
            classIds.count == width * height,
            confidence.count == width * height
        else {
            return (0, 0)
        }

        let normalizedX = (point.x - frame.minX) / frame.width
        let normalizedY = (point.y - frame.minY) / frame.height
        let x = min(max(Int((normalizedX * CGFloat(width - 1)).rounded()), 0), width - 1)
        let y = min(max(Int((normalizedY * CGFloat(height - 1)).rounded()), 0), height - 1)
        let index = y * width + x
        return (classIds[index], confidence[index])
    }
}

private extension ARKitLipMask {
    func alphaValue(at point: CGPoint) -> UInt8 {
        guard
            frame.width > 1,
            frame.height > 1,
            frame.contains(point),
            alpha.count == width * height
        else {
            return 0
        }

        let normalizedX = min(max((point.x - frame.minX) / frame.width, 0), 1)
        let normalizedY = min(max((point.y - frame.minY) / frame.height, 0), 1)
        let sampleX = normalizedX * CGFloat(width - 1)
        let sampleY = normalizedY * CGFloat(height - 1)
        let x0 = min(max(Int(floor(sampleX)), 0), width - 1)
        let y0 = min(max(Int(floor(sampleY)), 0), height - 1)
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = sampleX - CGFloat(x0)
        let ty = sampleY - CGFloat(y0)

        let a00 = CGFloat(alpha[y0 * width + x0])
        let a10 = CGFloat(alpha[y0 * width + x1])
        let a01 = CGFloat(alpha[y1 * width + x0])
        let a11 = CGFloat(alpha[y1 * width + x1])
        let top = a00 + ((a10 - a00) * tx)
        let bottom = a01 + ((a11 - a01) * tx)
        return UInt8(min(max(top + ((bottom - top) * ty), 0), 255))
    }
}

final class ARKitLipParsingModel: @unchecked Sendable {
    private static let inputSize = 256
    private static let classCount = 9
    private static let upperLipClass = 5
    private static let lowerLipClass = 6
    private static let innerMouthClass = 7

    private let model: MLModel?
    private let inputName: String?
    private let outputName: String?
    private let resourceName: String

    var isReady: Bool {
        model != nil && inputName != nil
    }

    init(resourceName: String = "MakeupParsingSmall") {
        self.resourceName = resourceName
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            configuration.allowLowPrecisionAccumulationOnGPU = true
            let modelURL = try Self.loadableModelURL(resourceName: resourceName)
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

    func makeRequest(
        pixelBuffer: CVPixelBuffer,
        roi: CGRect,
        arFrame: ARFrame,
        viewportSize: CGSize
    ) -> ARKitLipParsingRequest? {
        guard isReady, viewportSize.width > 1, viewportSize.height > 1 else { return nil }

        let transform = arFrame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        return ARKitLipParsingRequest(
            pixelBuffer: pixelBuffer,
            roi: roi,
            topLeft: Self.imagePoint(from: CGPoint(x: roi.minX, y: roi.minY), viewportSize: viewportSize, transform: transform),
            topRight: Self.imagePoint(from: CGPoint(x: roi.maxX, y: roi.minY), viewportSize: viewportSize, transform: transform),
            bottomLeft: Self.imagePoint(from: CGPoint(x: roi.minX, y: roi.maxY), viewportSize: viewportSize, transform: transform),
            bottomRight: Self.imagePoint(from: CGPoint(x: roi.maxX, y: roi.maxY), viewportSize: viewportSize, transform: transform)
        )
    }

    func mask(request: ARKitLipParsingRequest) -> ARKitLipMask? {
        guard let logits = logits(request: request) else { return nil }
        return makeMask(from: logits, frame: request.roi)
    }

    func faceMask(request: ARKitLipParsingRequest) -> ARKitFaceParsingMask? {
        guard let logits = logits(request: request) else { return nil }
        return makeFaceMask(from: logits, frame: request.roi)
    }

    private func logits(request: ARKitLipParsingRequest) -> MLMultiArray? {
        guard let model, let inputName else { return nil }
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

            return logits
        } catch {
            return nil
        }
    }

    private static func loadableModelURL(resourceName: String) throws -> URL {
        let compiledURLs = [
            Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc"),
            Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc", subdirectory: "Models")
        ].compactMap { $0 }

        if let compiledURL = compiledURLs.first {
            return compiledURL
        }

        let packageURLs = [
            Bundle.main.url(forResource: resourceName, withExtension: "mlpackage"),
            Bundle.main.url(forResource: resourceName, withExtension: "mlpackage", subdirectory: "Models")
        ].compactMap { $0 }

        if let packageURL = packageURLs.first {
            return try MLModel.compileModel(at: packageURL)
        }

        throw CocoaError(.fileNoSuchFile)
    }

    private static func imagePoint(
        from viewPoint: CGPoint,
        viewportSize: CGSize,
        transform: CGAffineTransform
    ) -> CGPoint {
        let viewNormalized = CGPoint(
            x: viewPoint.x / max(viewportSize.width, 1),
            y: viewPoint.y / max(viewportSize.height, 1)
        )
        let imagePoint = viewNormalized.applying(transform)
        return CGPoint(
            x: min(max(imagePoint.x, 0), 1),
            y: min(max(imagePoint.y, 0), 1)
        )
    }

    private func makeInputArray(request: ARKitLipParsingRequest) throws -> MLMultiArray {
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

        guard let sampler = ARKitPixelSampler(pixelBuffer: request.pixelBuffer), strides.count == 4 else {
            throw CocoaError(.coderInvalidValue)
        }

        sampler.fillNormalizedRGB(
            into: pointer,
            strides: strides,
            inputWidth: Self.inputSize,
            inputHeight: Self.inputSize,
            topLeft: request.topLeft,
            topRight: request.topRight,
            bottomLeft: request.bottomLeft,
            bottomRight: request.bottomRight,
            mean: mean,
            std: std
        )

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

    private func makeMask(from logits: MLMultiArray, frame: CGRect) -> ARKitLipMask? {
        var timing: LipDecodeTiming?
        return makeMask(from: logits, frame: frame, timing: &timing)
    }

    private func makeMask(
        from logits: MLMultiArray,
        frame: CGRect,
        timing: inout LipDecodeTiming?
    ) -> ARKitLipMask? {
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
                var bestOtherScore = mouthScore

                for channel in 0..<Self.classCount where channel != Self.upperLipClass && channel != Self.lowerLipClass {
                    bestOtherScore = max(bestOtherScore, value(in: logits, strides: strides, channel: channel, y: y, x: x))
                }

                let margin = lipScore - bestOtherScore
                guard lipScore > mouthScore, margin > -0.08 else { continue }

                let opacity = smoothstep((margin + 0.18) / 0.42)
                guard opacity > 0.12 else { continue }

                alpha[y * width + x] = UInt8(min(max(opacity * 235, 0), 235))
                lipPixelCount += 1
            }
        }

        let coverage = CGFloat(lipPixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.003, coverage < 0.34 else { return nil }

        alpha = softlyExpanded(alpha: alpha, width: width, height: height, radius: 1)
        alpha = blurred(alpha: alpha, width: width, height: height, radius: 1)
        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    private func makeFaceMask(from logits: MLMultiArray, frame: CGRect) -> ARKitFaceParsingMask? {
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 4, shape[1] >= Self.classCount else { return nil }

        let height = shape[2]
        let width = shape[3]
        guard width > 0, height > 0 else { return nil }

        let strides = logits.strides.map { $0.intValue }
        guard strides.count == 4 else { return nil }

        var classIds = [UInt8](repeating: 0, count: width * height)
        var confidence = [UInt8](repeating: 0, count: width * height)
        var activePixelCount = 0

        for y in 0..<height {
            for x in 0..<width {
                var bestClass = 0
                var bestScore = -Float.greatestFiniteMagnitude
                var secondBestScore = -Float.greatestFiniteMagnitude

                for channel in 0..<Self.classCount {
                    let score = value(in: logits, strides: strides, channel: channel, y: y, x: x)
                    if score > bestScore {
                        secondBestScore = bestScore
                        bestScore = score
                        bestClass = channel
                    } else if score > secondBestScore {
                        secondBestScore = score
                    }
                }

                guard bestClass > 0 else { continue }

                let margin = bestScore - secondBestScore
                let pixelConfidence = smoothstep((margin + 0.05) / 0.45)
                guard pixelConfidence > 0.14 else { continue }

                let index = y * width + x
                classIds[index] = UInt8(min(bestClass, 255))
                confidence[index] = UInt8(min(max(pixelConfidence * 255, 0), 255))
                activePixelCount += 1
            }
        }

        let coverage = CGFloat(activePixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.02, coverage < 0.82 else { return nil }

        return ARKitFaceParsingMask(
            frame: frame,
            width: width,
            height: height,
            classIds: classIds,
            confidence: confidence
        )
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
        if array.dataType == .float16 {
            let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            return Float(pointer[offset])
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
                for sampleY in max(0, y - radius)...min(height - 1, y + radius) {
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

private struct LipDecodeTiming {
    var rawMS: Double = 0
    var postMS: Double = 0
    var outputDataType: String
    var cropCoverage: CGFloat?
    var sampleStep: Int = 1
    var failureReason: String?
}

enum ARKitLipRefinementOutputKind {
    case semanticLogits
    case softAlpha
    case sdfAlpha
}

enum ARKitLipSoftAlphaDecoder {
    static func normalizedAlpha(from value: Float) -> Float {
        if value >= 0, value <= 1 {
            return value
        }
        return Float(1 / (1 + exp(Double(-value))))
    }

    static func alphaByte(from value: Float) -> UInt8 {
        let normalized = normalizedAlpha(from: value)
        return UInt8(min(max(normalized * 255, 0), 255))
    }
}

enum ARKitLipTemplatePrior {
    static func canonical(width: Int, height: Int) -> [Float] {
        guard width > 0, height > 0 else { return [] }

        let centerX = Float(width - 1) * 0.5
        let splitY = Float(height - 1) * 0.51
        let radiusX = max(Float(width) * 0.34, 2)
        let upperRadiusY = max(Float(height) * 0.18, 2)
        let lowerRadiusY = max(Float(height) * 0.23, 2)
        let upperCenterY = splitY - Float(height) * 0.085
        let lowerCenterY = splitY + Float(height) * 0.115

        var mask = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            let fy = Float(y)
            for x in 0..<width {
                let fx = Float(x)
                let normalizedX = pow((fx - centerX) / radiusX, 2)
                let upper = normalizedX + pow((fy - upperCenterY) / upperRadiusY, 2) <= 1
                    && fy <= splitY + upperRadiusY * 0.24
                let lower = normalizedX + pow((fy - lowerCenterY) / lowerRadiusY, 2) <= 1
                    && fy >= splitY - lowerRadiusY * 0.22
                mask[y * width + x] = upper || lower
            }
        }

        guard mask.contains(true) else { return [Float](repeating: 0, count: width * height) }

        let distanceToLip = distance(to: mask, width: width, height: height)
        let backgroundMask = mask.map { !$0 }
        let distanceToBackground = distance(to: backgroundMask, width: width, height: height)
        let featherRadius: Float = 3
        var template = [Float](repeating: 0, count: width * height)
        for index in template.indices {
            let signedDistance = distanceToBackground[index] - distanceToLip[index]
            template[index] = smoothstep((signedDistance + featherRadius) / (featherRadius * 2))
        }
        return template
    }

    private static func distance(to mask: [Bool], width: Int, height: Int) -> [Float] {
        guard mask.count == width * height, mask.contains(true) else {
            return [Float](repeating: 1000, count: max(width * height, 0))
        }

        let large = Double(width * width + height * height + 1)
        var grid = [Double](repeating: large, count: width * height)
        for index in mask.indices where mask[index] {
            grid[index] = 0
        }

        var vertical = [Double](repeating: 0, count: width * height)
        for x in 0..<width {
            var column = [Double](repeating: 0, count: height)
            for y in 0..<height {
                column[y] = grid[y * width + x]
            }
            let transformed = edt1D(column)
            for y in 0..<height {
                vertical[y * width + x] = transformed[y]
            }
        }

        var squared = [Double](repeating: 0, count: width * height)
        for y in 0..<height {
            var row = [Double](repeating: 0, count: width)
            for x in 0..<width {
                row[x] = vertical[y * width + x]
            }
            let transformed = edt1D(row)
            for x in 0..<width {
                squared[y * width + x] = transformed[x]
            }
        }

        return squared.map { Float(sqrt($0)) }
    }

    private static func edt1D(_ values: [Double]) -> [Double] {
        let count = values.count
        guard count > 0 else { return [] }

        var positions = [Int](repeating: 0, count: count)
        var boundaries = [Double](repeating: 0, count: count + 1)
        var output = [Double](repeating: 0, count: count)

        var active = 0
        positions[0] = 0
        boundaries[0] = -.infinity
        boundaries[1] = .infinity

        if count > 1 {
            for query in 1..<count {
                var intersection: Double
                repeat {
                    let previous = positions[active]
                    intersection = ((values[query] + Double(query * query)) - (values[previous] + Double(previous * previous)))
                        / Double(2 * query - 2 * previous)
                    if intersection <= boundaries[active] {
                        active -= 1
                    }
                } while intersection <= boundaries[active]

                active += 1
                positions[active] = query
                boundaries[active] = intersection
                boundaries[active + 1] = .infinity
            }
        }

        active = 0
        for query in 0..<count {
            while boundaries[active + 1] < Double(query) {
                active += 1
            }
            let nearest = positions[active]
            output[query] = Double((query - nearest) * (query - nearest)) + values[nearest]
        }
        return output
    }

    private static func smoothstep(_ value: Float) -> Float {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }
}

final class ARKitLipRefinementModel: @unchecked Sendable {
    private static let defaultInputWidth = 192
    private static let defaultInputHeight = 128
    private static let defaultInputChannels = 3
    private static let classCount = 4
    private static let upperLipClass = 1
    private static let lowerLipClass = 2
    private static let innerMouthClass = 3
    private static let realtimeMaskPostprocessEnabled = false
    private static let realtimeMaskFeatherEnabled = false

    private let model: MLModel?
    private let inputName: String?
    private let outputName: String?
    private let inputWidth: Int
    private let inputHeight: Int
    private let inputChannels: Int
    private let templatePrior: [Float]?
    private let resourceName: String
    private let outputKind: ARKitLipRefinementOutputKind

    var isReady: Bool {
        model != nil && inputName != nil
    }

    init(resourceName: String = "LipParsingSmall", outputKind: ARKitLipRefinementOutputKind = .semanticLogits) {
        self.resourceName = resourceName
        self.outputKind = outputKind
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            configuration.allowLowPrecisionAccumulationOnGPU = true
            let modelURL = try Self.loadableModelURL(resourceName: resourceName)
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            self.model = model
            self.inputName = model.modelDescription.inputDescriptionsByName.keys.first
            self.outputName = model.modelDescription.outputDescriptionsByName.keys.first
            let inputShape = Self.inputShape(from: model)
            self.inputWidth = inputShape.width
            self.inputHeight = inputShape.height
            self.inputChannels = inputShape.channels
            self.templatePrior = inputShape.channels >= 4
                ? ARKitLipTemplatePrior.canonical(width: inputShape.width, height: inputShape.height)
                : nil
        } catch {
            self.model = nil
            self.inputName = nil
            self.outputName = nil
            self.inputWidth = Self.defaultInputWidth
            self.inputHeight = Self.defaultInputHeight
            self.inputChannels = Self.defaultInputChannels
            self.templatePrior = nil
        }
    }

    func makeRequest(
        pixelBuffer: CVPixelBuffer,
        roi: CGRect,
        arFrame: ARFrame,
        viewportSize: CGSize
    ) -> ARKitLipParsingRequest? {
        guard isReady, viewportSize.width > 1, viewportSize.height > 1 else { return nil }

        let transform = arFrame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        return ARKitLipParsingRequest(
            pixelBuffer: pixelBuffer,
            roi: roi,
            topLeft: Self.imagePoint(from: CGPoint(x: roi.minX, y: roi.minY), viewportSize: viewportSize, transform: transform),
            topRight: Self.imagePoint(from: CGPoint(x: roi.maxX, y: roi.minY), viewportSize: viewportSize, transform: transform),
            bottomLeft: Self.imagePoint(from: CGPoint(x: roi.minX, y: roi.maxY), viewportSize: viewportSize, transform: transform),
            bottomRight: Self.imagePoint(from: CGPoint(x: roi.maxX, y: roi.maxY), viewportSize: viewportSize, transform: transform)
        )
    }

    func mask(request: ARKitLipParsingRequest) -> ARKitLipMask? {
        guard let logits = logits(request: request) else { return nil }
        return makeMask(from: logits, frame: request.roi)
    }

    func timedMask(request: ARKitLipParsingRequest) -> (
        mask: ARKitLipMask?,
        inputMS: Double,
        predictionMS: Double,
        decodeMS: Double,
        decodeRawMS: Double,
        decodePostMS: Double,
        outputDataType: String
    ) {
        guard let model, let inputName else {
            return (nil, 0, 0, 0, 0, 0, "none")
        }

        let inputStart = CACurrentMediaTime()
        let input: MLMultiArray
        do {
            input = try makeInputArray(request: request)
        } catch {
            return (nil, (CACurrentMediaTime() - inputStart) * 1000, 0, 0, 0, 0, "none")
        }
        let inputMS = (CACurrentMediaTime() - inputStart) * 1000

        let predictionStart = CACurrentMediaTime()
        let logits: MLMultiArray
        do {
            let provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(multiArray: input)]
            )
            let prediction = try model.prediction(from: provider)
            guard
                let outputName = outputName ?? prediction.featureNames.first,
                let output = prediction.featureValue(for: outputName)?.multiArrayValue
            else {
                return (nil, inputMS, (CACurrentMediaTime() - predictionStart) * 1000, 0, 0, 0, "none")
            }
            logits = output
        } catch {
            return (nil, inputMS, (CACurrentMediaTime() - predictionStart) * 1000, 0, 0, 0, "none")
        }
        let predictionMS = (CACurrentMediaTime() - predictionStart) * 1000

        let decodeStart = CACurrentMediaTime()
        var decodeTiming: LipDecodeTiming? = LipDecodeTiming(outputDataType: Self.dataTypeLabel(logits.dataType))
        let mask: ARKitLipMask?
        switch outputKind {
        case .semanticLogits:
            mask = makeMask(from: logits, frame: request.roi, timing: &decodeTiming)
        case .softAlpha:
            mask = makeSoftAlphaMask(from: logits, frame: request.roi, timing: &decodeTiming)
        case .sdfAlpha:
            mask = makeSDFAlphaMask(from: logits, frame: request.roi, timing: &decodeTiming)
        }
        let decodeMS = (CACurrentMediaTime() - decodeStart) * 1000
        return (
            mask,
            inputMS,
            predictionMS,
            decodeMS,
            decodeTiming?.rawMS ?? 0,
            decodeTiming?.postMS ?? 0,
            decodeTiming?.outputDataType ?? "unknown"
        )
    }

    private func logits(request: ARKitLipParsingRequest) -> MLMultiArray? {
        guard let model, let inputName else { return nil }
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

            return logits
        } catch {
            return nil
        }
    }

    private static func loadableModelURL(resourceName: String) throws -> URL {
        let compiledURLs = [
            Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc"),
            Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc", subdirectory: "Models")
        ].compactMap { $0 }

        if let compiledURL = compiledURLs.first {
            return compiledURL
        }

        let packageURLs = [
            Bundle.main.url(forResource: resourceName, withExtension: "mlpackage"),
            Bundle.main.url(forResource: resourceName, withExtension: "mlpackage", subdirectory: "Models")
        ].compactMap { $0 }

        if let packageURL = packageURLs.first {
            return try MLModel.compileModel(at: packageURL)
        }

        throw CocoaError(.fileNoSuchFile)
    }

    private static func inputShape(from model: MLModel) -> (width: Int, height: Int, channels: Int) {
        guard
            let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
            let description = model.modelDescription.inputDescriptionsByName[inputName],
            let shape = description.multiArrayConstraint?.shape.map({ $0.intValue }),
            shape.count == 4
        else {
            return (defaultInputWidth, defaultInputHeight, defaultInputChannels)
        }

        return (
            width: max(shape[3], 1),
            height: max(shape[2], 1),
            channels: max(shape[1], defaultInputChannels)
        )
    }

    private static func dataTypeLabel(_ dataType: MLMultiArrayDataType) -> String {
        switch dataType {
        case .float32:
            return "f32"
        case .float16:
            return "f16"
        case .double:
            return "f64"
        case .int32:
            return "i32"
        default:
            return "dt\(dataType.rawValue)"
        }
    }

    private static func imagePoint(
        from viewPoint: CGPoint,
        viewportSize: CGSize,
        transform: CGAffineTransform
    ) -> CGPoint {
        let viewNormalized = CGPoint(
            x: viewPoint.x / max(viewportSize.width, 1),
            y: viewPoint.y / max(viewportSize.height, 1)
        )
        let imagePoint = viewNormalized.applying(transform)
        return CGPoint(
            x: min(max(imagePoint.x, 0), 1),
            y: min(max(imagePoint.y, 0), 1)
        )
    }

    private func makeInputArray(request: ARKitLipParsingRequest) throws -> MLMultiArray {
        let input = try MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: inputChannels),
                NSNumber(value: inputHeight),
                NSNumber(value: inputWidth)
            ],
            dataType: .float32
        )
        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let strides = input.strides.map { $0.intValue }
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        CVPixelBufferLockBaseAddress(request.pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(request.pixelBuffer, .readOnly) }

        guard let sampler = ARKitPixelSampler(pixelBuffer: request.pixelBuffer), strides.count == 4 else {
            throw CocoaError(.coderInvalidValue)
        }

        sampler.fillNormalizedRGB(
            into: pointer,
            strides: strides,
            inputWidth: inputWidth,
            inputHeight: inputHeight,
            topLeft: request.topLeft,
            topRight: request.topRight,
            bottomLeft: request.bottomLeft,
            bottomRight: request.bottomRight,
            mean: mean,
            std: std
        )

        if inputChannels > 3 {
            let channelStride = strides[1]
            let rowStride = strides[2]
            let columnStride = strides[3]
            for channel in 3..<inputChannels {
                let channelOffset = channel * channelStride
                for y in 0..<inputHeight {
                    let rowOffset = y * rowStride
                    for x in 0..<inputWidth {
                        let outputOffset = channelOffset + rowOffset + x * columnStride
                        if channel == 3, let templatePrior {
                            pointer[outputOffset] = templatePrior[y * inputWidth + x]
                        } else {
                            pointer[outputOffset] = 0
                        }
                    }
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

    private func makeMask(from logits: MLMultiArray, frame: CGRect) -> ARKitLipMask? {
        var timing: LipDecodeTiming?
        return makeMask(from: logits, frame: frame, timing: &timing)
    }

    private func makeMask(
        from logits: MLMultiArray,
        frame: CGRect,
        timing: inout LipDecodeTiming?
    ) -> ARKitLipMask? {
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 4, shape[1] >= Self.classCount else { return nil }

        let height = shape[2]
        let width = shape[3]
        guard width > 0, height > 0 else { return nil }

        let strides = logits.strides.map { $0.intValue }
        guard strides.count == 4 else { return nil }
        timing?.outputDataType = Self.dataTypeLabel(logits.dataType)

        if logits.dataType == .float32 {
            let scores = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
            return makeMaskFromFloat32Scores(
                scores,
                strides: strides,
                width: width,
                height: height,
                frame: frame,
                timing: &timing
            )
        }
        if logits.dataType == .float16 {
            let scores = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
            return makeMaskFromFloat16Scores(
                scores,
                strides: strides,
                width: width,
                height: height,
                frame: frame,
                timing: &timing
            )
        }

        let rawStart = CACurrentMediaTime()
        var alpha = [UInt8](repeating: 0, count: width * height)
        var lipPixelCount = 0

        for y in 0..<height {
            for x in 0..<width {
                let backgroundScore = value(in: logits, strides: strides, channel: 0, y: y, x: x)
                let upperScore = value(in: logits, strides: strides, channel: Self.upperLipClass, y: y, x: x)
                let lowerScore = value(in: logits, strides: strides, channel: Self.lowerLipClass, y: y, x: x)
                let innerScore = value(in: logits, strides: strides, channel: Self.innerMouthClass, y: y, x: x)

                let maxScore = max(backgroundScore, upperScore, lowerScore, innerScore)
                let backgroundExp = exp(Double(backgroundScore - maxScore))
                let upperExp = exp(Double(upperScore - maxScore))
                let lowerExp = exp(Double(lowerScore - maxScore))
                let innerExp = exp(Double(innerScore - maxScore))
                let denominator = max(backgroundExp + upperExp + lowerExp + innerExp, 0.000001)

                let upperProbability = Float(upperExp / denominator)
                let lowerProbability = Float(lowerExp / denominator)
                let innerProbability = Float(innerExp / denominator)
                let lipProbability = upperProbability + lowerProbability

                let isLowerLip = lowerScore >= upperScore
                let lipScore = max(upperScore, lowerScore)
                let nonLipScore = max(backgroundScore, innerScore)
                let index = y * width + x

                let innerMouthIsDominant = innerProbability > 0.46 && innerScore > lipScore - 0.05
                guard !innerMouthIsDominant else {
                    continue
                }

                let gapOpacity = smoothstep((lipScore - nonLipScore + 0.10) / 0.48)
                let probabilityOpacity = smoothstep((lipProbability - 0.22) / 0.48)
                var opacity = max(gapOpacity, probabilityOpacity)
                let verticalProgress = (Float(y) + 0.5) / Float(height)
                let margin = lipScore - nonLipScore
                guard
                    verticalProgress > 0.24,
                    verticalProgress > 0.36 || margin > 0.10,
                    verticalProgress > 0.30 || margin > 0.22
                else {
                    continue
                }

                if isLowerLip {
                    opacity *= 1.34
                    if verticalProgress > 0.44, lipProbability > 0.17 {
                        opacity = max(opacity, smoothstep((lipProbability - 0.14) / 0.50) * 1.02)
                    }
                    if verticalProgress > 0.56, lipProbability > 0.14 {
                        opacity = max(opacity, smoothstep((lipProbability - 0.11) / 0.54) * 0.88)
                    }
                } else {
                    opacity *= 0.94
                    if verticalProgress < 0.24 {
                        opacity *= 0.55
                    }
                }

                let minimumLipProbability: Float = isLowerLip && verticalProgress > 0.42 ? 0.15 : 0.20
                let minimumOpacity: Float = isLowerLip && verticalProgress > 0.42 ? 0.045 : 0.075
                guard lipProbability > minimumLipProbability, opacity > minimumOpacity else { continue }

                let alphaValue = UInt8(min(max(opacity * 245, 0), 245))
                alpha[index] = max(alpha[index], alphaValue)
                if isLowerLip, verticalProgress > 0.42 {
                    addLowerLipPadding(
                        to: &alpha,
                        width: width,
                        height: height,
                        x: x,
                        y: y,
                        alphaValue: alphaValue,
                        verticalProgress: verticalProgress
                    )
                }
                lipPixelCount += 1
            }
        }
        timing?.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let coverage = CGFloat(lipPixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.010, coverage < 0.52 else { return nil }

        let postStart = CACurrentMediaTime()
        if Self.realtimeMaskPostprocessEnabled {
            alpha = removeTinyComponents(alpha: alpha, width: width, height: height, threshold: 20)
            alpha = blurred(alpha: alpha, width: width, height: height, radius: 1)
        } else if Self.realtimeMaskFeatherEnabled {
            alpha = feathered(alpha: alpha, width: width, height: height)
        }
        timing?.postMS = (CACurrentMediaTime() - postStart) * 1000
        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    private func makeMaskFromFloat32Scores(
        _ scores: UnsafePointer<Float>,
        strides: [Int],
        width: Int,
        height: Int,
        frame: CGRect,
        timing: inout LipDecodeTiming?
    ) -> ARKitLipMask? {
        let rawStart = CACurrentMediaTime()
        var alpha = [UInt8](repeating: 0, count: width * height)
        var lipPixelCount = 0

        let backgroundBase = 0
        let upperBase = Self.upperLipClass * strides[1]
        let lowerBase = Self.lowerLipClass * strides[1]
        let innerBase = Self.innerMouthClass * strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]

        for y in 0..<height {
            let rowOffset = y * rowStride
            let verticalProgress = (Float(y) + 0.5) / Float(height)

            for x in 0..<width {
                let offset = rowOffset + x * columnStride
                let backgroundScore = scores[backgroundBase + offset]
                let upperScore = scores[upperBase + offset]
                let lowerScore = scores[lowerBase + offset]
                let innerScore = scores[innerBase + offset]
                let isLowerLip = lowerScore >= upperScore
                let lipScore = isLowerLip ? lowerScore : upperScore
                let nonLipScore = max(backgroundScore, innerScore)
                let index = y * width + x

                let innerMouthIsDominant = innerScore > lipScore - 0.04 && innerScore > backgroundScore + 0.04
                guard !innerMouthIsDominant else { continue }

                let margin = lipScore - nonLipScore
                guard
                    verticalProgress > 0.24,
                    verticalProgress > 0.36 || margin > 0.10,
                    verticalProgress > 0.30 || margin > 0.22
                else {
                    continue
                }
                var opacity = smoothstep((margin + 0.16) / 0.58)

                if isLowerLip {
                    opacity *= 1.34
                    if verticalProgress > 0.44, margin > -0.20 {
                        opacity = max(opacity, smoothstep((margin + 0.30) / 0.72) * 1.02)
                    }
                    if verticalProgress > 0.56, margin > -0.24 {
                        opacity = max(opacity, smoothstep((margin + 0.34) / 0.78) * 0.90)
                    }
                } else {
                    opacity *= 0.94
                    if verticalProgress < 0.24 {
                        opacity *= 0.55
                    }
                }

                let backgroundSlack: Float = isLowerLip && verticalProgress > 0.42 ? -0.32 : -0.20
                let minimumOpacity: Float = isLowerLip && verticalProgress > 0.42 ? 0.045 : 0.075
                guard lipScore > backgroundScore + backgroundSlack, opacity > minimumOpacity else { continue }

                let alphaValue = UInt8(min(max(opacity * 245, 0), 245))
                alpha[index] = max(alpha[index], alphaValue)
                if isLowerLip, verticalProgress > 0.42 {
                    addLowerLipPadding(
                        to: &alpha,
                        width: width,
                        height: height,
                        x: x,
                        y: y,
                        alphaValue: alphaValue,
                        verticalProgress: verticalProgress
                    )
                }
                lipPixelCount += 1
            }
        }
        timing?.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let coverage = CGFloat(lipPixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.010, coverage < 0.52 else { return nil }

        let postStart = CACurrentMediaTime()
        if Self.realtimeMaskPostprocessEnabled {
            alpha = removeTinyComponents(alpha: alpha, width: width, height: height, threshold: 20)
            alpha = blurred(alpha: alpha, width: width, height: height, radius: 1)
        } else if Self.realtimeMaskFeatherEnabled {
            alpha = feathered(alpha: alpha, width: width, height: height)
        }
        timing?.postMS = (CACurrentMediaTime() - postStart) * 1000
        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    private func makeMaskFromFloat16Scores(
        _ scores: UnsafePointer<Float16>,
        strides: [Int],
        width: Int,
        height: Int,
        frame: CGRect,
        timing: inout LipDecodeTiming?
    ) -> ARKitLipMask? {
        let rawStart = CACurrentMediaTime()
        var alpha = [UInt8](repeating: 0, count: width * height)
        var lipPixelCount = 0

        let backgroundBase = 0
        let upperBase = Self.upperLipClass * strides[1]
        let lowerBase = Self.lowerLipClass * strides[1]
        let innerBase = Self.innerMouthClass * strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]

        for y in 0..<height {
            let rowOffset = y * rowStride
            let verticalProgress = (Float(y) + 0.5) / Float(height)

            for x in 0..<width {
                let offset = rowOffset + x * columnStride
                let backgroundScore = Float(scores[backgroundBase + offset])
                let upperScore = Float(scores[upperBase + offset])
                let lowerScore = Float(scores[lowerBase + offset])
                let innerScore = Float(scores[innerBase + offset])
                let isLowerLip = lowerScore >= upperScore
                let lipScore = isLowerLip ? lowerScore : upperScore
                let nonLipScore = max(backgroundScore, innerScore)
                let index = y * width + x

                let innerMouthIsDominant = innerScore > lipScore - 0.04 && innerScore > backgroundScore + 0.04
                guard !innerMouthIsDominant else { continue }

                let margin = lipScore - nonLipScore
                guard
                    verticalProgress > 0.24,
                    verticalProgress > 0.36 || margin > 0.10,
                    verticalProgress > 0.30 || margin > 0.22
                else {
                    continue
                }
                var opacity = smoothstep((margin + 0.16) / 0.58)

                if isLowerLip {
                    opacity *= 1.34
                    if verticalProgress > 0.44, margin > -0.20 {
                        opacity = max(opacity, smoothstep((margin + 0.30) / 0.72) * 1.02)
                    }
                    if verticalProgress > 0.56, margin > -0.24 {
                        opacity = max(opacity, smoothstep((margin + 0.34) / 0.78) * 0.90)
                    }
                } else {
                    opacity *= 0.94
                    if verticalProgress < 0.24 {
                        opacity *= 0.55
                    }
                }

                let backgroundSlack: Float = isLowerLip && verticalProgress > 0.42 ? -0.32 : -0.20
                let minimumOpacity: Float = isLowerLip && verticalProgress > 0.42 ? 0.045 : 0.075
                guard lipScore > backgroundScore + backgroundSlack, opacity > minimumOpacity else { continue }

                let alphaValue = UInt8(min(max(opacity * 245, 0), 245))
                alpha[index] = max(alpha[index], alphaValue)
                if isLowerLip, verticalProgress > 0.42 {
                    addLowerLipPadding(
                        to: &alpha,
                        width: width,
                        height: height,
                        x: x,
                        y: y,
                        alphaValue: alphaValue,
                        verticalProgress: verticalProgress
                    )
                }
                lipPixelCount += 1
            }
        }
        timing?.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let coverage = CGFloat(lipPixelCount) / CGFloat(max(width * height, 1))
        guard coverage > 0.010, coverage < 0.52 else { return nil }

        let postStart = CACurrentMediaTime()
        if Self.realtimeMaskPostprocessEnabled {
            alpha = removeTinyComponents(alpha: alpha, width: width, height: height, threshold: 20)
            alpha = blurred(alpha: alpha, width: width, height: height, radius: 1)
        } else if Self.realtimeMaskFeatherEnabled {
            alpha = feathered(alpha: alpha, width: width, height: height)
        }
        timing?.postMS = (CACurrentMediaTime() - postStart) * 1000
        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
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
        if array.dataType == .float16 {
            let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            return Float(pointer[offset])
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

    private func addLowerLipPadding(
        to alpha: inout [UInt8],
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        alphaValue: UInt8,
        verticalProgress: Float
    ) {
        guard width > 1, height > 1, y < height - 1 else { return }

        let strongLower = verticalProgress > 0.54
        let primaryFactor: Float = strongLower ? 0.62 : 0.50
        let sideFactor: Float = strongLower ? 0.46 : 0.36
        let secondRowFactor: Float = strongLower ? 0.34 : 0.24

        func write(_ targetX: Int, _ targetY: Int, factor: Float, cap: UInt8) {
            guard targetX >= 0, targetX < width, targetY >= 0, targetY < height else { return }
            let index = targetY * width + targetX
            let value = UInt8(min(max(Int(Float(alphaValue) * factor), 0), Int(cap)))
            if value > alpha[index] {
                alpha[index] = value
            }
        }

        write(x, y + 1, factor: primaryFactor, cap: strongLower ? 168 : 138)
        write(x - 1, y + 1, factor: sideFactor, cap: 118)
        write(x + 1, y + 1, factor: sideFactor, cap: 118)

        if y < height - 2 {
            write(x, y + 2, factor: secondRowFactor, cap: strongLower ? 96 : 72)
        }
    }

    private func removeTinyComponents(alpha: [UInt8], width: Int, height: Int, threshold: UInt8) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        var output = alpha
        var visited = [Bool](repeating: false, count: alpha.count)
        var stack: [Int] = []
        var component: [Int] = []
        let minimumPixels = max(20, Int(CGFloat(width * height) * 0.00042))

        for start in alpha.indices where !visited[start] && alpha[start] > threshold {
            stack.removeAll(keepingCapacity: true)
            component.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true

            while let index = stack.popLast() {
                component.append(index)
                let x = index % width
                let y = index / width

                if x > 0 {
                    let neighbor = y * width + x - 1
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
                    }
                }
                if x < width - 1 {
                    let neighbor = y * width + x + 1
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
                    }
                }
                if y > 0 {
                    let neighbor = (y - 1) * width + x
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
                    }
                }
                if y < height - 1 {
                    let neighbor = (y + 1) * width + x
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold { stack.append(neighbor) }
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

    private func feathered(alpha: [UInt8], width: Int, height: Int) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else {
            return alpha
        }

        var output = alpha
        for y in 0..<height {
            let verticalProgress = Float(y) / Float(max(height - 1, 1))
            let neighborFactor: Float
            if verticalProgress < 0.24 {
                neighborFactor = 0.16
            } else if verticalProgress < 0.34 {
                neighborFactor = 0.30
            } else if verticalProgress < 0.46 {
                neighborFactor = 0.48
            } else {
                neighborFactor = 0.64
            }

            for x in 0..<width {
                let index = y * width + x
                let current = alpha[index]
                var neighborMax = current

                if x > 0 {
                    neighborMax = max(neighborMax, alpha[index - 1])
                }
                if x < width - 1 {
                    neighborMax = max(neighborMax, alpha[index + 1])
                }
                if y > 0 {
                    let above = index - width
                    neighborMax = max(neighborMax, alpha[above])
                    if x > 0 {
                        neighborMax = max(neighborMax, alpha[above - 1])
                    }
                    if x < width - 1 {
                        neighborMax = max(neighborMax, alpha[above + 1])
                    }
                }
                if y < height - 1 {
                    let below = index + width
                    neighborMax = max(neighborMax, alpha[below])
                    if x > 0 {
                        neighborMax = max(neighborMax, alpha[below - 1])
                    }
                    if x < width - 1 {
                        neighborMax = max(neighborMax, alpha[below + 1])
                    }
                }

                guard neighborMax > current else { continue }
                let featherCap = verticalProgress > 0.42 ? 154 : 118
                let feather = UInt8(min(max(Int(Float(neighborMax) * neighborFactor), 0), featherCap))
                output[index] = max(current, feather)
            }
        }

        return output
    }

    private func classAwareExpanded(
        alpha: [UInt8],
        classes: [UInt8],
        innerMouthBlockers: [Bool],
        width: Int,
        height: Int
    ) -> [UInt8] {
        guard
            width > 1,
            height > 1,
            alpha.count == width * height,
            classes.count == alpha.count,
            innerMouthBlockers.count == alpha.count
        else {
            return alpha
        }

        var output = alpha
        for y in 0..<height {
            let verticalProgress = CGFloat(y) / CGFloat(max(height - 1, 1))
            for x in 0..<width where alpha[y * width + x] == 0 {
                let index = y * width + x
                guard !innerMouthBlockers[index] else { continue }

                var lowerNeighborMax = 0
                var upperNeighborMax = 0
                for sampleY in max(0, y - 2)...min(height - 1, y + 2) {
                    for sampleX in max(0, x - 1)...min(width - 1, x + 1) {
                        let sampleIndex = sampleY * width + sampleX
                        if classes[sampleIndex] == UInt8(Self.lowerLipClass) {
                            lowerNeighborMax = max(lowerNeighborMax, Int(alpha[sampleIndex]))
                        } else if classes[sampleIndex] == UInt8(Self.upperLipClass) {
                            upperNeighborMax = max(upperNeighborMax, Int(alpha[sampleIndex]))
                        }
                    }
                }

                if verticalProgress > 0.42, lowerNeighborMax > 44 {
                    output[index] = UInt8(min(max(CGFloat(lowerNeighborMax) * 0.66, 0), 156))
                } else if verticalProgress > 0.23, upperNeighborMax > 72 {
                    output[index] = UInt8(min(max(CGFloat(upperNeighborMax) * 0.26, 0), 76))
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

private extension ARKitLipRefinementModel {
    func makeSoftAlphaMask(
        from output: MLMultiArray,
        frame: CGRect,
        timing: inout LipDecodeTiming?
    ) -> ARKitLipMask? {
        let shape = output.shape.map { $0.intValue }
        let strides = output.strides.map { $0.intValue }
        let width: Int
        let height: Int
        let rowStride: Int
        let columnStride: Int

        if shape.count == 4, strides.count == 4, shape[1] >= 1 {
            height = shape[2]
            width = shape[3]
            rowStride = strides[2]
            columnStride = strides[3]
        } else if shape.count == 3, strides.count == 3 {
            height = shape[1]
            width = shape[2]
            rowStride = strides[1]
            columnStride = strides[2]
        } else {
            return nil
        }

        guard width > 0, height > 0 else { return nil }
        timing?.outputDataType = Self.softAlphaDataTypeLabel(output.dataType)

        let rawStart = CACurrentMediaTime()
        var alpha = [UInt8](repeating: 0, count: width * height)
        var activePixelCount = 0
        var maximumAlpha: UInt8 = 0

        switch output.dataType {
        case .float32:
            let values = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            for y in 0..<height {
                let rowOffset = y * rowStride
                for x in 0..<width {
                    let alphaValue = ARKitLipSoftAlphaDecoder.alphaByte(from: values[rowOffset + x * columnStride])
                    alpha[y * width + x] = alphaValue
                    if alphaValue > 10 { activePixelCount += 1 }
                    maximumAlpha = max(maximumAlpha, alphaValue)
                }
            }
        case .float16:
            let values = output.dataPointer.bindMemory(to: Float16.self, capacity: output.count)
            for y in 0..<height {
                let rowOffset = y * rowStride
                for x in 0..<width {
                    let alphaValue = ARKitLipSoftAlphaDecoder.alphaByte(from: Float(values[rowOffset + x * columnStride]))
                    alpha[y * width + x] = alphaValue
                    if alphaValue > 10 { activePixelCount += 1 }
                    maximumAlpha = max(maximumAlpha, alphaValue)
                }
            }
        default:
            for y in 0..<height {
                for x in 0..<width {
                    let indices: [NSNumber]
                    if shape.count == 4 {
                        indices = [0, 0, NSNumber(value: y), NSNumber(value: x)]
                    } else {
                        indices = [0, NSNumber(value: y), NSNumber(value: x)]
                    }
                    let alphaValue = ARKitLipSoftAlphaDecoder.alphaByte(from: output[indices].floatValue)
                    alpha[y * width + x] = alphaValue
                    if alphaValue > 10 { activePixelCount += 1 }
                    maximumAlpha = max(maximumAlpha, alphaValue)
                }
            }
        }

        timing?.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let coverage = CGFloat(activePixelCount) / CGFloat(max(width * height, 1))
        timing?.cropCoverage = coverage
        guard coverage > 0.010, coverage < 0.66, maximumAlpha >= 16 else { return nil }

        if #available(iOS 14.0, *), let smoothed = LipMaskSmoother.smoothMask(width: width, height: height, alpha: alpha) {
            alpha = smoothed
        }

        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    func makeSDFAlphaMask(
        from output: MLMultiArray,
        frame: CGRect,
        timing: inout LipDecodeTiming?
    ) -> ARKitLipMask? {
        let shape = output.shape.map { $0.intValue }
        let strides = output.strides.map { $0.intValue }
        guard shape.count == 4, strides.count == 4, shape[1] >= 2 else {
            return makeSoftAlphaMask(from: output, frame: frame, timing: &timing)
        }

        let height = shape[2]
        let width = shape[3]
        let alphaChannelOffset = 0
        let sdfChannelOffset = strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]
        guard width > 0, height > 0 else { return nil }

        timing?.outputDataType = "\(Self.softAlphaDataTypeLabel(output.dataType))+sdf"

        let rawStart = CACurrentMediaTime()
        var alpha = [UInt8](repeating: 0, count: width * height)
        var activePixelCount = 0
        var maximumAlpha: UInt8 = 0

        switch output.dataType {
        case .float32:
            let values = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            for y in 0..<height {
                let alphaRowOffset = alphaChannelOffset + y * rowStride
                let sdfRowOffset = sdfChannelOffset + y * rowStride
                for x in 0..<width {
                    let offset = x * columnStride
                    let alphaValue = sdfAlphaByte(
                        alphaRaw: values[alphaRowOffset + offset],
                        sdfRaw: values[sdfRowOffset + offset]
                    )
                    alpha[y * width + x] = alphaValue
                    if alphaValue > 10 { activePixelCount += 1 }
                    maximumAlpha = max(maximumAlpha, alphaValue)
                }
            }
        case .float16:
            let values = output.dataPointer.bindMemory(to: Float16.self, capacity: output.count)
            for y in 0..<height {
                let alphaRowOffset = alphaChannelOffset + y * rowStride
                let sdfRowOffset = sdfChannelOffset + y * rowStride
                for x in 0..<width {
                    let offset = x * columnStride
                    let alphaValue = sdfAlphaByte(
                        alphaRaw: Float(values[alphaRowOffset + offset]),
                        sdfRaw: Float(values[sdfRowOffset + offset])
                    )
                    alpha[y * width + x] = alphaValue
                    if alphaValue > 10 { activePixelCount += 1 }
                    maximumAlpha = max(maximumAlpha, alphaValue)
                }
            }
        default:
            for y in 0..<height {
                for x in 0..<width {
                    let alphaRaw = output[
                        [
                            NSNumber(value: 0),
                            NSNumber(value: 0),
                            NSNumber(value: y),
                            NSNumber(value: x)
                        ]
                    ].floatValue
                    let sdfRaw = output[
                        [
                            NSNumber(value: 0),
                            NSNumber(value: 1),
                            NSNumber(value: y),
                            NSNumber(value: x)
                        ]
                    ].floatValue
                    let alphaValue = sdfAlphaByte(alphaRaw: alphaRaw, sdfRaw: sdfRaw)
                    alpha[y * width + x] = alphaValue
                    if alphaValue > 10 { activePixelCount += 1 }
                    maximumAlpha = max(maximumAlpha, alphaValue)
                }
            }
        }

        timing?.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let coverage = CGFloat(activePixelCount) / CGFloat(max(width * height, 1))
        timing?.cropCoverage = coverage
        guard coverage > 0.010, coverage < 0.66, maximumAlpha >= 16 else { return nil }

        if #available(iOS 14.0, *), let smoothed = LipMaskSmoother.smoothMask(width: width, height: height, alpha: alpha) {
            alpha = smoothed
        }

        return ARKitLipMask(frame: frame, width: width, height: height, alpha: alpha)
    }

    private func sdfAlphaByte(alphaRaw: Float, sdfRaw: Float) -> UInt8 {
        let confidence = ARKitLipSoftAlphaDecoder.normalizedAlpha(from: alphaRaw)
        let signedDistance = Float(tanh(Double(sdfRaw)))

        // SDF targets are positive inside the lip and negative outside. Keep
        // the distance field in charge of the border, with alpha as a guard.
        let sdfFeather: Float = 0.11
        let distanceAlpha = smoothstep((signedDistance + sdfFeather) / (sdfFeather * 2))
        let confidenceGate = smoothstep((confidence - 0.015) / 0.22)
        let combined = distanceAlpha * confidenceGate

        return UInt8(min(max(combined * 255, 0), 255))
    }

    static func softAlphaDataTypeLabel(_ dataType: MLMultiArrayDataType) -> String {
        switch dataType {
        case .float32:
            return "f32"
        case .float16:
            return "f16"
        case .double:
            return "f64"
        case .int32:
            return "i32"
        default:
            return "dt\(dataType.rawValue)"
        }
    }
}

struct ARKitUVLipStudentTriangle: @unchecked Sendable {
    let uv0: SIMD2<Float>
    let uv1: SIMD2<Float>
    let uv2: SIMD2<Float>
    let image0: CGPoint
    let image1: CGPoint
    let image2: CGPoint
}

struct ARKitUVLipStudentRequest: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let triangles: [ARKitUVLipStudentTriangle]
}

final class ARKitUVLipStudentModel: @unchecked Sendable {
    private static let sourceUVSize = 256
    private static let outputUVSize = 256
    private static let cropMinX = 64
    private static let cropMinY = 128
    private static let cropMaxX = 184
    private static let cropMaxY = 216
    private static let outputCropPaddingX = 0
    private static let outputCropPaddingTop = 1
    private static let outputCropPaddingBottom = 1
    private static let defaultInputWidth = 96
    private static let defaultInputHeight = 72
    private static let fastDecodeInputThreshold = 96
    private static let classCount = 4
    private static let upperLipClass = 1
    private static let lowerLipClass = 2
    private static let innerMouthClass = 3

    private let model: MLModel?
    private let inputName: String?
    private let outputName: String?
    private let inputWidth: Int
    private let inputHeight: Int

    var isReady: Bool {
        model != nil && inputName != nil
    }

    init(resourceName: String = "ARKitUVLipAlphaSmall") {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            configuration.allowLowPrecisionAccumulationOnGPU = true
            let modelURL = try Self.loadableModelURL(resourceName: resourceName)
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            self.model = model
            self.inputName = model.modelDescription.inputDescriptionsByName.keys.first
            self.outputName = model.modelDescription.outputDescriptionsByName.keys.first
            let inputSize = Self.inputSize(from: model)
            self.inputWidth = inputSize.width
            self.inputHeight = inputSize.height
        } catch {
            self.model = nil
            self.inputName = nil
            self.outputName = nil
            self.inputWidth = Self.defaultInputWidth
            self.inputHeight = Self.defaultInputHeight
        }
    }

    func makeRequest(
        pixelBuffer: CVPixelBuffer,
        arFrame: ARFrame,
        faceAnchor: ARFaceAnchor,
        viewportSize: CGSize
    ) -> ARKitUVLipStudentRequest? {
        guard isReady, viewportSize.width > 1, viewportSize.height > 1 else { return nil }

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

        let inverseDisplayTransform = arFrame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        var screenPoints = [CGPoint](repeating: .zero, count: vertices.count)
        var imagePoints = [CGPoint](repeating: .zero, count: vertices.count)
        var validPoints = [Bool](repeating: false, count: vertices.count)

        for index in vertices.indices {
            let vertex = vertices[index]
            let worldPoint4 = simd_mul(faceAnchor.transform, simd_float4(vertex.x, vertex.y, vertex.z, 1))
            let worldPoint = simd_float3(worldPoint4.x, worldPoint4.y, worldPoint4.z)
            let screenPoint = arFrame.camera.projectPoint(
                worldPoint,
                orientation: .portrait,
                viewportSize: viewportSize
            )

            guard
                screenPoint.x.isFinite,
                screenPoint.y.isFinite,
                screenPoint.x > -viewportSize.width * 0.35,
                screenPoint.x < viewportSize.width * 1.35,
                screenPoint.y > -viewportSize.height * 0.35,
                screenPoint.y < viewportSize.height * 1.35
            else {
                continue
            }

            screenPoints[index] = screenPoint
            imagePoints[index] = Self.imagePoint(from: screenPoint, viewportSize: viewportSize, transform: inverseDisplayTransform)
            validPoints[index] = true
        }

        let cropBounds = CGRect(
            x: Self.cropMinX,
            y: Self.cropMinY,
            width: Self.cropMaxX - Self.cropMinX,
            height: Self.cropMaxY - Self.cropMinY
        )
        var triangles: [ARKitUVLipStudentTriangle] = []
        triangles.reserveCapacity(120)

        for offset in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let i0 = Int(triangleIndices[offset])
            let i1 = Int(triangleIndices[offset + 1])
            let i2 = Int(triangleIndices[offset + 2])

            guard
                vertices.indices.contains(i0),
                vertices.indices.contains(i1),
                vertices.indices.contains(i2),
                validPoints[i0],
                validPoints[i1],
                validPoints[i2]
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

            let p0 = SIMD2<Float>(uv0.x * Float(Self.sourceUVSize - 1), uv0.y * Float(Self.sourceUVSize - 1))
            let p1 = SIMD2<Float>(uv1.x * Float(Self.sourceUVSize - 1), uv1.y * Float(Self.sourceUVSize - 1))
            let p2 = SIMD2<Float>(uv2.x * Float(Self.sourceUVSize - 1), uv2.y * Float(Self.sourceUVSize - 1))
            let uvTriangleBounds = CGRect(
                x: CGFloat(min(p0.x, p1.x, p2.x)),
                y: CGFloat(min(p0.y, p1.y, p2.y)),
                width: CGFloat(max(p0.x, p1.x, p2.x) - min(p0.x, p1.x, p2.x)),
                height: CGFloat(max(p0.y, p1.y, p2.y) - min(p0.y, p1.y, p2.y))
            ).insetBy(dx: -1, dy: -1)
            guard uvTriangleBounds.intersects(cropBounds) else { continue }

            let screenArea = Self.signedArea(screenPoints[i0], screenPoints[i1], screenPoints[i2])
            guard screenArea < -0.0001, abs(screenArea) > 0.04 else { continue }

            triangles.append(
                ARKitUVLipStudentTriangle(
                    uv0: p0,
                    uv1: p1,
                    uv2: p2,
                    image0: imagePoints[i0],
                    image1: imagePoints[i1],
                    image2: imagePoints[i2]
                )
            )
        }

        guard triangles.count >= 16 else { return nil }
        return ARKitUVLipStudentRequest(pixelBuffer: pixelBuffer, triangles: triangles)
    }

    func timedLipAlphaMask(request: ARKitUVLipStudentRequest) -> (
        mask: UVLipAlphaMask?,
        inputMS: Double,
        predictionMS: Double,
        decodeMS: Double,
        decodeRawMS: Double,
        decodePostMS: Double,
        outputDataType: String,
        decodeInfo: String
    ) {
        guard let model, let inputName else {
            return (nil, 0, 0, 0, 0, 0, "none", "noModel")
        }

        let inputStart = CACurrentMediaTime()
        let input: MLMultiArray
        do {
            input = try makeInputArray(request: request)
        } catch {
            return (nil, (CACurrentMediaTime() - inputStart) * 1000, 0, 0, 0, 0, "none", "inputFail")
        }
        let inputMS = (CACurrentMediaTime() - inputStart) * 1000

        let predictionStart = CACurrentMediaTime()
        let logits: MLMultiArray
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
            let prediction = try model.prediction(from: provider)
            guard
                let outputName = outputName ?? prediction.featureNames.first,
                let output = prediction.featureValue(for: outputName)?.multiArrayValue
            else {
                return (nil, inputMS, (CACurrentMediaTime() - predictionStart) * 1000, 0, 0, 0, "none", "noOutput")
            }
            logits = output
        } catch {
            return (nil, inputMS, (CACurrentMediaTime() - predictionStart) * 1000, 0, 0, 0, "none", "predFail")
        }
        let predictionMS = (CACurrentMediaTime() - predictionStart) * 1000

        let decodeStart = CACurrentMediaTime()
        var timing = LipDecodeTiming(outputDataType: Self.dataTypeLabel(logits.dataType))
        let mask = makeLipAlphaMask(from: logits, timing: &timing)
        let decodeMS = (CACurrentMediaTime() - decodeStart) * 1000
        let coverageText: String
        if let cropCoverage = timing.cropCoverage {
            coverageText = String(format: " crop%.1f%%", cropCoverage * 100)
        } else {
            coverageText = ""
        }
        let failureText = timing.failureReason.map { " \($0)" } ?? ""
        let decodeInfo = "s\(timing.sampleStep)\(coverageText)\(failureText)"
        return (
            mask,
            inputMS,
            predictionMS,
            decodeMS,
            timing.rawMS,
            timing.postMS,
            timing.outputDataType,
            decodeInfo
        )
    }

    func timedLipColorSupportMask(
        request: ARKitUVLipStudentRequest,
        candidate: UVLipAlphaMask
    ) -> (mask: UVLipAlphaMask?, ms: Double, info: String) {
        let start = CACurrentMediaTime()
        let result = makeLipColorSupportMask(request: request, candidate: candidate)
        let ms = (CACurrentMediaTime() - start) * 1000

        guard let mask = result.mask else {
            return (nil, ms, result.info)
        }

        let info = String(
            format: "%@ %.2f%% sep%.3f",
            result.info,
            mask.nonzeroCoverage * 100,
            result.separation
        )
        return (mask, ms, info)
    }

    private static func loadableModelURL(resourceName: String) throws -> URL {
        let compiledURLs = [
            Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc"),
            Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc", subdirectory: "Models")
        ].compactMap { $0 }
        if let compiledURL = compiledURLs.first {
            return compiledURL
        }

        let packageURLs = [
            Bundle.main.url(forResource: resourceName, withExtension: "mlpackage"),
            Bundle.main.url(forResource: resourceName, withExtension: "mlpackage", subdirectory: "Models")
        ].compactMap { $0 }
        if let packageURL = packageURLs.first {
            return try MLModel.compileModel(at: packageURL)
        }

        throw CocoaError(.fileNoSuchFile)
    }

    private static func inputSize(from model: MLModel) -> (width: Int, height: Int) {
        guard
            let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
            let description = model.modelDescription.inputDescriptionsByName[inputName],
            let shape = description.multiArrayConstraint?.shape.map({ $0.intValue }),
            shape.count == 4
        else {
            return (defaultInputWidth, defaultInputHeight)
        }

        return (
            width: max(shape[3], 1),
            height: max(shape[2], 1)
        )
    }

    private static func dataTypeLabel(_ dataType: MLMultiArrayDataType) -> String {
        switch dataType {
        case .float32:
            return "f32"
        case .float16:
            return "f16"
        case .double:
            return "f64"
        case .int32:
            return "i32"
        default:
            return "dt\(dataType.rawValue)"
        }
    }

    private static func imagePoint(
        from viewPoint: CGPoint,
        viewportSize: CGSize,
        transform: CGAffineTransform
    ) -> CGPoint {
        let viewNormalized = CGPoint(
            x: viewPoint.x / max(viewportSize.width, 1),
            y: viewPoint.y / max(viewportSize.height, 1)
        )
        let imagePoint = viewNormalized.applying(transform)
        return CGPoint(
            x: min(max(imagePoint.x, 0), 1),
            y: min(max(imagePoint.y, 0), 1)
        )
    }

    private func makeInputArray(request: ARKitUVLipStudentRequest) throws -> MLMultiArray {
        let input = try MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: 3),
                NSNumber(value: inputHeight),
                NSNumber(value: inputWidth)
            ],
            dataType: .float32
        )
        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let strides = input.strides.map { $0.intValue }
        guard strides.count == 4 else { throw CocoaError(.coderInvalidValue) }

        CVPixelBufferLockBaseAddress(request.pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(request.pixelBuffer, .readOnly) }
        guard let sampler = ARKitPixelSampler(pixelBuffer: request.pixelBuffer) else {
            throw CocoaError(.coderInvalidValue)
        }

        let pixelCount = inputWidth * inputHeight
        var red = [Float](repeating: 0, count: pixelCount)
        var green = [Float](repeating: 0, count: pixelCount)
        var blue = [Float](repeating: 0, count: pixelCount)
        var filled = [Bool](repeating: false, count: pixelCount)

        for triangle in request.triangles {
            rasterizeInputTriangle(
                triangle,
                sampler: sampler,
                red: &red,
                green: &green,
                blue: &blue,
                filled: &filled
            )
        }

        var filledCount = 0
        var meanRed: Float = 0
        var meanGreen: Float = 0
        var meanBlue: Float = 0
        for index in 0..<pixelCount where filled[index] {
            filledCount += 1
            meanRed += red[index]
            meanGreen += green[index]
            meanBlue += blue[index]
        }
        if filledCount > 0 {
            meanRed /= Float(filledCount)
            meanGreen /= Float(filledCount)
            meanBlue /= Float(filledCount)
        } else {
            meanRed = 0.45
            meanGreen = 0.42
            meanBlue = 0.40
        }

        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        let channelStride = strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]
        for y in 0..<inputHeight {
            let rowOffset = y * rowStride
            for x in 0..<inputWidth {
                let pixelIndex = y * inputWidth + x
                let outputOffset = rowOffset + x * columnStride
                let sampleRed = filled[pixelIndex] ? red[pixelIndex] : meanRed
                let sampleGreen = filled[pixelIndex] ? green[pixelIndex] : meanGreen
                let sampleBlue = filled[pixelIndex] ? blue[pixelIndex] : meanBlue
                pointer[outputOffset] = (sampleRed - mean[0]) / std[0]
                pointer[channelStride + outputOffset] = (sampleGreen - mean[1]) / std[1]
                pointer[channelStride * 2 + outputOffset] = (sampleBlue - mean[2]) / std[2]
            }
        }

        return input
    }

    private func makeLipColorSupportMask(
        request: ARKitUVLipStudentRequest,
        candidate: UVLipAlphaMask
    ) -> (mask: UVLipAlphaMask?, info: String, separation: CGFloat) {
        guard
            candidate.width == Self.outputUVSize,
            candidate.height == Self.outputUVSize,
            candidate.alpha.count == Self.outputUVSize * Self.outputUVSize
        else {
            return (nil, "color size", 0)
        }

        CVPixelBufferLockBaseAddress(request.pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(request.pixelBuffer, .readOnly) }
        guard let sampler = ARKitPixelSampler(pixelBuffer: request.pixelBuffer) else {
            return (nil, "color sampler", 0)
        }

        let size = Self.outputUVSize
        let count = size * size
        let cropBounds = CGRect(
            x: Self.cropMinX,
            y: Self.cropMinY,
            width: Self.cropMaxX - Self.cropMinX,
            height: Self.cropMaxY - Self.cropMinY
        ).insetBy(dx: -2, dy: -2)
        var features = [CGFloat](repeating: 0, count: count)
        var filled = [Bool](repeating: false, count: count)

        for triangle in request.triangles {
            rasterizeColorTriangle(
                triangle,
                sampler: sampler,
                features: &features,
                filled: &filled,
                cropBounds: cropBounds
            )
        }

        let activeIndices = candidate.alpha.indices.filter { candidate.alpha[$0] > 32 && filled[$0] }
        guard activeIndices.count > 80 else {
            return (nil, "color noBase", 0)
        }

        let minX = activeIndices.map { $0 % size }.min() ?? Self.cropMinX
        let maxX = activeIndices.map { $0 % size }.max() ?? Self.cropMaxX
        let minY = activeIndices.map { $0 / size }.min() ?? Self.cropMinY
        let maxY = activeIndices.map { $0 / size }.max() ?? Self.cropMaxY
        let ringMinX = max(Self.cropMinX, minX - 8)
        let ringMaxX = min(Self.cropMaxX - 1, maxX + 8)
        let ringMinY = max(Self.cropMinY, minY - 8)
        let ringMaxY = min(Self.cropMaxY - 1, maxY + 8)

        var lipFeatures: [CGFloat] = []
        lipFeatures.reserveCapacity(activeIndices.count)
        var skinFeatures: [CGFloat] = []
        skinFeatures.reserveCapacity(512)

        for index in activeIndices where candidate.alpha[index] > 128 {
            lipFeatures.append(features[index])
        }

        if ringMinX <= ringMaxX, ringMinY <= ringMaxY {
            for y in ringMinY...ringMaxY {
                for x in ringMinX...ringMaxX {
                    let index = y * size + x
                    guard filled[index], candidate.alpha[index] <= 16 else { continue }
                    skinFeatures.append(features[index])
                }
            }
        }

        guard lipFeatures.count > 48 else {
            return (nil, "color noLip", 0)
        }

        let lipReference = Self.percentile(lipFeatures, fraction: 0.52)
        let skinReference = skinFeatures.count > 48
            ? Self.percentile(skinFeatures, fraction: 0.62)
            : lipReference - 0.045
        let separation = lipReference - skinReference
        let adaptiveThreshold = max(skinReference + max(separation * 0.28, 0.018), lipReference - 0.075)

        var support = [UInt8](repeating: 0, count: count)
        for index in candidate.alpha.indices where filled[index] {
            let amount = Self.smoothstep((features[index] - adaptiveThreshold) / 0.085)
            guard amount > 0 else { continue }
            support[index] = UInt8(min(max(amount * 255, 0), 255))
        }

        var supportedBase = 0
        for index in activeIndices where support[index] > 44 {
            supportedBase += 1
        }
        let supportedRatio = CGFloat(supportedBase) / CGFloat(max(activeIndices.count, 1))
        guard supportedRatio > 0.16 else {
            return (nil, String(format: "color weak%.2f", supportedRatio), separation)
        }

        return (UVLipAlphaMask(width: size, height: size, alpha: support), "color ok", separation)
    }

    private func rasterizeColorTriangle(
        _ triangle: ARKitUVLipStudentTriangle,
        sampler: ARKitPixelSampler,
        features: inout [CGFloat],
        filled: inout [Bool],
        cropBounds: CGRect
    ) {
        let size = Self.outputUVSize
        let p0 = CGPoint(x: CGFloat(triangle.uv0.x), y: CGFloat(triangle.uv0.y))
        let p1 = CGPoint(x: CGFloat(triangle.uv1.x), y: CGFloat(triangle.uv1.y))
        let p2 = CGPoint(x: CGFloat(triangle.uv2.x), y: CGFloat(triangle.uv2.y))
        let bounds = CGRect(
            x: min(p0.x, p1.x, p2.x),
            y: min(p0.y, p1.y, p2.y),
            width: max(p0.x, p1.x, p2.x) - min(p0.x, p1.x, p2.x),
            height: max(p0.y, p1.y, p2.y) - min(p0.y, p1.y, p2.y)
        ).insetBy(dx: -1, dy: -1)
        guard bounds.intersects(cropBounds) else { return }

        let minX = min(max(Int(floor(max(bounds.minX, cropBounds.minX))), 0), size - 1)
        let maxX = min(max(Int(ceil(min(bounds.maxX, cropBounds.maxX))), 0), size - 1)
        let minY = min(max(Int(floor(max(bounds.minY, cropBounds.minY))), 0), size - 1)
        let maxY = min(max(Int(ceil(min(bounds.maxY, cropBounds.maxY))), 0), size - 1)
        guard minX <= maxX, minY <= maxY else { return }

        let denominator = ((p1.y - p2.y) * (p0.x - p2.x)) + ((p2.x - p1.x) * (p0.y - p2.y))
        guard abs(denominator) > 0.0001 else { return }

        for y in minY...maxY {
            for x in minX...maxX {
                let point = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
                let w0 = (((p1.y - p2.y) * (point.x - p2.x)) + ((p2.x - p1.x) * (point.y - p2.y))) / denominator
                let w1 = (((p2.y - p0.y) * (point.x - p2.x)) + ((p0.x - p2.x) * (point.y - p2.y))) / denominator
                let w2 = 1 - w0 - w1

                guard w0 >= -0.001, w1 >= -0.001, w2 >= -0.001 else { continue }

                let imagePoint = CGPoint(
                    x: triangle.image0.x * w0 + triangle.image1.x * w1 + triangle.image2.x * w2,
                    y: triangle.image0.y * w0 + triangle.image1.y * w1 + triangle.image2.y * w2
                )
                guard let color = sampler.sample(normalizedX: imagePoint.x, normalizedY: imagePoint.y) else { continue }

                let index = y * size + x
                features[index] = Self.lipColorFeature(color)
                filled[index] = true
            }
        }
    }

    private func rasterizeInputTriangle(
        _ triangle: ARKitUVLipStudentTriangle,
        sampler: ARKitPixelSampler,
        red: inout [Float],
        green: inout [Float],
        blue: inout [Float],
        filled: inout [Bool]
    ) {
        let p0 = inputPoint(from: triangle.uv0)
        let p1 = inputPoint(from: triangle.uv1)
        let p2 = inputPoint(from: triangle.uv2)
        let minX = min(max(Int(floor(min(p0.x, p1.x, p2.x))), 0), inputWidth - 1)
        let maxX = min(max(Int(ceil(max(p0.x, p1.x, p2.x))), 0), inputWidth - 1)
        let minY = min(max(Int(floor(min(p0.y, p1.y, p2.y))), 0), inputHeight - 1)
        let maxY = min(max(Int(ceil(max(p0.y, p1.y, p2.y))), 0), inputHeight - 1)
        guard minX <= maxX, minY <= maxY else { return }

        let denominator = ((p1.y - p2.y) * (p0.x - p2.x)) + ((p2.x - p1.x) * (p0.y - p2.y))
        guard abs(denominator) > 0.0001 else { return }

        for y in minY...maxY {
            for x in minX...maxX {
                let point = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
                let w0 = (((p1.y - p2.y) * (point.x - p2.x)) + ((p2.x - p1.x) * (point.y - p2.y))) / denominator
                let w1 = (((p2.y - p0.y) * (point.x - p2.x)) + ((p0.x - p2.x) * (point.y - p2.y))) / denominator
                let w2 = 1 - w0 - w1
                guard w0 >= -0.001, w1 >= -0.001, w2 >= -0.001 else { continue }

                let imageX = triangle.image0.x * w0 + triangle.image1.x * w1 + triangle.image2.x * w2
                let imageY = triangle.image0.y * w0 + triangle.image1.y * w1 + triangle.image2.y * w2
                guard let color = sampler.sample(normalizedX: imageX, normalizedY: imageY) else { continue }

                let index = y * inputWidth + x
                red[index] = Float(color.red)
                green[index] = Float(color.green)
                blue[index] = Float(color.blue)
                filled[index] = true
            }
        }
    }

    private func inputPoint(from uvPixel: SIMD2<Float>) -> CGPoint {
        let cropWidth = CGFloat(Self.cropMaxX - Self.cropMinX)
        let cropHeight = CGFloat(Self.cropMaxY - Self.cropMinY)
        return CGPoint(
            x: ((CGFloat(uvPixel.x) - CGFloat(Self.cropMinX)) / cropWidth) * CGFloat(inputWidth),
            y: ((CGFloat(uvPixel.y) - CGFloat(Self.cropMinY)) / cropHeight) * CGFloat(inputHeight)
        )
    }

    private func makeLipAlphaMask(from logits: MLMultiArray, timing: inout LipDecodeTiming) -> UVLipAlphaMask? {
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 4 else { return nil }
        let channelCount = shape[1]
        guard channelCount == 1 || channelCount >= Self.classCount else { return nil }

        let height = shape[2]
        let width = shape[3]
        guard width > 0, height > 0 else { return nil }

        let strides = logits.strides.map { $0.intValue }
        guard strides.count == 4 else { return nil }
        timing.outputDataType = Self.dataTypeLabel(logits.dataType)

        if logits.dataType == .float32 {
            let scores = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
            if channelCount == 1 {
                let outputsAreProbabilities = singleChannelOutputLooksLikeProbabilityMap(
                    scores,
                    strides: strides,
                    width: width,
                    height: height
                )
                return makeLipAlphaMaskFromFloat32AlphaLogits(
                    scores,
                    strides: strides,
                    width: width,
                    height: height,
                    outputsAreProbabilities: outputsAreProbabilities,
                    timing: &timing
                )
            }
            return makeLipAlphaMaskFromFloat32Scores(scores, strides: strides, width: width, height: height, timing: &timing)
        }
        if logits.dataType == .float16 {
            let scores = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
            if channelCount == 1 {
                let outputsAreProbabilities = singleChannelOutputLooksLikeProbabilityMap(
                    scores,
                    strides: strides,
                    width: width,
                    height: height
                )
                return makeLipAlphaMaskFromFloat16AlphaLogits(
                    scores,
                    strides: strides,
                    width: width,
                    height: height,
                    outputsAreProbabilities: outputsAreProbabilities,
                    timing: &timing
                )
            }
            return makeLipAlphaMaskFromFloat16Scores(scores, strides: strides, width: width, height: height, timing: &timing)
        }

        return nil
    }

    private func makeLipAlphaMaskFromFloat32Scores(
        _ scores: UnsafePointer<Float>,
        strides: [Int],
        width: Int,
        height: Int,
        timing: inout LipDecodeTiming
    ) -> UVLipAlphaMask? {
        let rawStart = CACurrentMediaTime()
        let cropAlpha = cropAlphaFromFloat32Scores(scores, strides: strides, width: width, height: height, timing: &timing)
        timing.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let postStart = CACurrentMediaTime()
        let mask = makeFullUVMask(
            fromCropAlpha: cropAlpha?.alpha,
            cropWidth: cropAlpha?.width ?? width,
            cropHeight: cropAlpha?.height ?? height
        )
        timing.postMS = (CACurrentMediaTime() - postStart) * 1000
        return mask
    }

    private func makeLipAlphaMaskFromFloat32AlphaLogits(
        _ scores: UnsafePointer<Float>,
        strides: [Int],
        width: Int,
        height: Int,
        outputsAreProbabilities: Bool,
        timing: inout LipDecodeTiming
    ) -> UVLipAlphaMask? {
        let rawStart = CACurrentMediaTime()
        let cropAlpha = cropAlphaFromFloat32AlphaLogits(
            scores,
            strides: strides,
            width: width,
            height: height,
            outputsAreProbabilities: outputsAreProbabilities,
            timing: &timing
        )
        timing.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let postStart = CACurrentMediaTime()
        let mask = makeFullUVMask(
            fromCropAlpha: cropAlpha?.alpha,
            cropWidth: cropAlpha?.width ?? width,
            cropHeight: cropAlpha?.height ?? height
        )
        timing.postMS = (CACurrentMediaTime() - postStart) * 1000
        return mask
    }

    private func makeLipAlphaMaskFromFloat16Scores(
        _ scores: UnsafePointer<Float16>,
        strides: [Int],
        width: Int,
        height: Int,
        timing: inout LipDecodeTiming
    ) -> UVLipAlphaMask? {
        let rawStart = CACurrentMediaTime()
        let cropAlpha = cropAlphaFromFloat16Scores(scores, strides: strides, width: width, height: height, timing: &timing)
        timing.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let postStart = CACurrentMediaTime()
        let mask = makeFullUVMask(
            fromCropAlpha: cropAlpha?.alpha,
            cropWidth: cropAlpha?.width ?? width,
            cropHeight: cropAlpha?.height ?? height
        )
        timing.postMS = (CACurrentMediaTime() - postStart) * 1000
        return mask
    }

    private func makeLipAlphaMaskFromFloat16AlphaLogits(
        _ scores: UnsafePointer<Float16>,
        strides: [Int],
        width: Int,
        height: Int,
        outputsAreProbabilities: Bool,
        timing: inout LipDecodeTiming
    ) -> UVLipAlphaMask? {
        let rawStart = CACurrentMediaTime()
        let cropAlpha = cropAlphaFromFloat16AlphaLogits(
            scores,
            strides: strides,
            width: width,
            height: height,
            outputsAreProbabilities: outputsAreProbabilities,
            timing: &timing
        )
        timing.rawMS = (CACurrentMediaTime() - rawStart) * 1000

        let postStart = CACurrentMediaTime()
        let mask = makeFullUVMask(
            fromCropAlpha: cropAlpha?.alpha,
            cropWidth: cropAlpha?.width ?? width,
            cropHeight: cropAlpha?.height ?? height
        )
        timing.postMS = (CACurrentMediaTime() - postStart) * 1000
        return mask
    }

    private func cropAlphaFromFloat32Scores(
        _ scores: UnsafePointer<Float>,
        strides: [Int],
        width: Int,
        height: Int,
        timing: inout LipDecodeTiming
    ) -> (alpha: [UInt8], width: Int, height: Int)? {
        let sampleStep = width > Self.fastDecodeInputThreshold ? 2 : 1
        timing.sampleStep = sampleStep
        let outputWidth = (width + sampleStep - 1) / sampleStep
        let outputHeight = (height + sampleStep - 1) / sampleStep
        var alpha = [UInt8](repeating: 0, count: outputWidth * outputHeight)
        var lipPixelCount = 0
        let channelStride = strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]

        for outputY in 0..<outputHeight {
            let y = min(outputY * sampleStep, height - 1)
            let rowOffset = y * rowStride
            let outputRowOffset = outputY * outputWidth
            for outputX in 0..<outputWidth {
                let x = min(outputX * sampleStep, width - 1)
                let offset = rowOffset + x * columnStride
                let value = lipAlphaValue(
                    backgroundScore: scores[offset],
                    upperScore: scores[Self.upperLipClass * channelStride + offset],
                    lowerScore: scores[Self.lowerLipClass * channelStride + offset],
                    innerScore: scores[Self.innerMouthClass * channelStride + offset]
                )
                if value > 0 {
                    alpha[outputRowOffset + outputX] = value
                    lipPixelCount += 1
                }
            }
        }

        guard let alpha = finalizeCropAlpha(alpha, lipPixelCount: lipPixelCount, width: outputWidth, height: outputHeight, expandEdges: false, timing: &timing) else {
            return nil
        }
        return (alpha, outputWidth, outputHeight)
    }

    private func cropAlphaFromFloat32AlphaLogits(
        _ scores: UnsafePointer<Float>,
        strides: [Int],
        width: Int,
        height: Int,
        outputsAreProbabilities: Bool,
        timing: inout LipDecodeTiming
    ) -> (alpha: [UInt8], width: Int, height: Int)? {
        let sampleStep = width > Self.fastDecodeInputThreshold ? 2 : 1
        timing.sampleStep = sampleStep
        let outputWidth = (width + sampleStep - 1) / sampleStep
        let outputHeight = (height + sampleStep - 1) / sampleStep
        var alpha = [UInt8](repeating: 0, count: outputWidth * outputHeight)
        var lipPixelCount = 0
        let rowStride = strides[2]
        let columnStride = strides[3]

        for outputY in 0..<outputHeight {
            let y = min(outputY * sampleStep, height - 1)
            let rowOffset = y * rowStride
            let outputRowOffset = outputY * outputWidth
            for outputX in 0..<outputWidth {
                let x = min(outputX * sampleStep, width - 1)
                let offset = rowOffset + x * columnStride
                let value = lipAlphaValue(alphaScore: scores[offset], isProbability: outputsAreProbabilities)
                alpha[outputRowOffset + outputX] = value
                if value > 18 {
                    lipPixelCount += 1
                }
            }
        }

        guard let alpha = finalizeCropAlpha(alpha, lipPixelCount: lipPixelCount, width: outputWidth, height: outputHeight, expandEdges: false, timing: &timing) else {
            return nil
        }
        return (alpha, outputWidth, outputHeight)
    }

    private func cropAlphaFromFloat16Scores(
        _ scores: UnsafePointer<Float16>,
        strides: [Int],
        width: Int,
        height: Int,
        timing: inout LipDecodeTiming
    ) -> (alpha: [UInt8], width: Int, height: Int)? {
        let sampleStep = width > Self.fastDecodeInputThreshold ? 2 : 1
        timing.sampleStep = sampleStep
        let outputWidth = (width + sampleStep - 1) / sampleStep
        let outputHeight = (height + sampleStep - 1) / sampleStep
        var alpha = [UInt8](repeating: 0, count: outputWidth * outputHeight)
        var lipPixelCount = 0
        let channelStride = strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]

        for outputY in 0..<outputHeight {
            let y = min(outputY * sampleStep, height - 1)
            let rowOffset = y * rowStride
            let outputRowOffset = outputY * outputWidth
            for outputX in 0..<outputWidth {
                let x = min(outputX * sampleStep, width - 1)
                let offset = rowOffset + x * columnStride
                let value = lipAlphaValue(
                    backgroundScore: Float(scores[offset]),
                    upperScore: Float(scores[Self.upperLipClass * channelStride + offset]),
                    lowerScore: Float(scores[Self.lowerLipClass * channelStride + offset]),
                    innerScore: Float(scores[Self.innerMouthClass * channelStride + offset])
                )
                if value > 0 {
                    alpha[outputRowOffset + outputX] = value
                    lipPixelCount += 1
                }
            }
        }

        guard let alpha = finalizeCropAlpha(alpha, lipPixelCount: lipPixelCount, width: outputWidth, height: outputHeight, expandEdges: false, timing: &timing) else {
            return nil
        }
        return (alpha, outputWidth, outputHeight)
    }

    private func cropAlphaFromFloat16AlphaLogits(
        _ scores: UnsafePointer<Float16>,
        strides: [Int],
        width: Int,
        height: Int,
        outputsAreProbabilities: Bool,
        timing: inout LipDecodeTiming
    ) -> (alpha: [UInt8], width: Int, height: Int)? {
        let sampleStep = width > Self.fastDecodeInputThreshold ? 2 : 1
        timing.sampleStep = sampleStep
        let outputWidth = (width + sampleStep - 1) / sampleStep
        let outputHeight = (height + sampleStep - 1) / sampleStep
        var alpha = [UInt8](repeating: 0, count: outputWidth * outputHeight)
        var lipPixelCount = 0
        let rowStride = strides[2]
        let columnStride = strides[3]

        for outputY in 0..<outputHeight {
            let y = min(outputY * sampleStep, height - 1)
            let rowOffset = y * rowStride
            let outputRowOffset = outputY * outputWidth
            for outputX in 0..<outputWidth {
                let x = min(outputX * sampleStep, width - 1)
                let offset = rowOffset + x * columnStride
                let value = lipAlphaValue(alphaScore: Float(scores[offset]), isProbability: outputsAreProbabilities)
                alpha[outputRowOffset + outputX] = value
                if value > 18 {
                    lipPixelCount += 1
                }
            }
        }

        guard let alpha = finalizeCropAlpha(alpha, lipPixelCount: lipPixelCount, width: outputWidth, height: outputHeight, expandEdges: false, timing: &timing) else {
            return nil
        }
        return (alpha, outputWidth, outputHeight)
    }

    private func finalizeCropAlpha(
        _ alpha: [UInt8],
        lipPixelCount: Int,
        width: Int,
        height: Int,
        expandEdges: Bool = true,
        timing: inout LipDecodeTiming
    ) -> [UInt8]? {
        let coverage = CGFloat(lipPixelCount) / CGFloat(max(width * height, 1))
        timing.cropCoverage = coverage
        guard coverage > 0.002 else {
            timing.failureReason = "lowCrop"
            return nil
        }
        guard coverage < 0.52 else {
            timing.failureReason = "highCrop"
            return nil
        }

        let cleaned = removeTinyComponents(alpha: alpha, width: width, height: height, threshold: 24)
        let edgeAdjusted = expandEdges ? expanded(alpha: cleaned, width: width, height: height, threshold: 24) : cleaned
        let smoothed = blurred(alpha: edgeAdjusted, width: width, height: height, radius: 1)
        return softClipped(alpha: smoothed)
    }

    @inline(__always)
    private func lipAlphaValue(
        backgroundScore: Float,
        upperScore: Float,
        lowerScore: Float,
        innerScore: Float
    ) -> UInt8 {
        let lipScore = upperScore > lowerScore ? upperScore : lowerScore
        guard lipScore > innerScore + 0.01 else { return 0 }

        let nonLipScore = backgroundScore > innerScore ? backgroundScore : innerScore
        let margin = lipScore - nonLipScore
        let opacity = smoothstep((margin + 0.08) / 0.42)
        guard opacity > 0.075 else { return 0 }

        return UInt8(min(max(opacity * 245, 0), 245))
    }

    @inline(__always)
    private func lipAlphaValue(alphaScore: Float, isProbability: Bool) -> UInt8 {
        let probability = isProbability
            ? min(max(alphaScore, 0), 1)
            : sigmoid(alphaScore)
        let opacity = smoothstep((probability - 0.16) / 0.70)
        guard opacity > 0.035 else { return 0 }
        return UInt8(min(max(opacity * 245, 0), 245))
    }

    private func singleChannelOutputLooksLikeProbabilityMap(
        _ scores: UnsafePointer<Float>,
        strides: [Int],
        width: Int,
        height: Int
    ) -> Bool {
        let rowStride = strides[2]
        let columnStride = strides[3]
        let sampleColumns = min(width, 9)
        let sampleRows = min(height, 9)
        var minimumValue = Float.greatestFiniteMagnitude
        var maximumValue = -Float.greatestFiniteMagnitude

        for sampleY in 0..<sampleRows {
            let y = min((sampleY * max(height - 1, 0)) / max(sampleRows - 1, 1), height - 1)
            let rowOffset = y * rowStride
            for sampleX in 0..<sampleColumns {
                let x = min((sampleX * max(width - 1, 0)) / max(sampleColumns - 1, 1), width - 1)
                let value = scores[rowOffset + x * columnStride]
                minimumValue = min(minimumValue, value)
                maximumValue = max(maximumValue, value)
            }
        }

        return minimumValue >= -0.001 && maximumValue <= 1.001
    }

    private func singleChannelOutputLooksLikeProbabilityMap(
        _ scores: UnsafePointer<Float16>,
        strides: [Int],
        width: Int,
        height: Int
    ) -> Bool {
        let rowStride = strides[2]
        let columnStride = strides[3]
        let sampleColumns = min(width, 9)
        let sampleRows = min(height, 9)
        var minimumValue = Float.greatestFiniteMagnitude
        var maximumValue = -Float.greatestFiniteMagnitude

        for sampleY in 0..<sampleRows {
            let y = min((sampleY * max(height - 1, 0)) / max(sampleRows - 1, 1), height - 1)
            let rowOffset = y * rowStride
            for sampleX in 0..<sampleColumns {
                let x = min((sampleX * max(width - 1, 0)) / max(sampleColumns - 1, 1), width - 1)
                let value = Float(scores[rowOffset + x * columnStride])
                minimumValue = min(minimumValue, value)
                maximumValue = max(maximumValue, value)
            }
        }

        return minimumValue >= -0.001 && maximumValue <= 1.001
    }

    private func makeFullUVMask(fromCropAlpha cropAlpha: [UInt8]?, cropWidth: Int, cropHeight: Int) -> UVLipAlphaMask? {
        guard
            let cropAlpha,
            cropWidth > 1,
            cropHeight > 1,
            cropAlpha.count == cropWidth * cropHeight
        else {
            return nil
        }

        let outputSize = Self.outputUVSize
        var fullAlpha = [UInt8](repeating: 0, count: outputSize * outputSize)
        let scale = CGFloat(outputSize) / CGFloat(Self.sourceUVSize)
        let outMinX = max(0, Int(floor(CGFloat(Self.cropMinX) * scale)) - Self.outputCropPaddingX)
        let outMaxX = min(outputSize, Int(ceil(CGFloat(Self.cropMaxX) * scale)) + Self.outputCropPaddingX)
        let outMinY = max(0, Int(floor(CGFloat(Self.cropMinY) * scale)) - Self.outputCropPaddingTop)
        let outMaxY = min(outputSize, Int(ceil(CGFloat(Self.cropMaxY) * scale)) + Self.outputCropPaddingBottom)
        let outCropWidth = max(outMaxX - outMinX, 1)
        let outCropHeight = max(outMaxY - outMinY, 1)

        for y in outMinY..<outMaxY {
            let v = (CGFloat(y - outMinY) + 0.5) / CGFloat(outCropHeight)
            let sourceY = v * CGFloat(cropHeight) - 0.5
            for x in outMinX..<outMaxX {
                let u = (CGFloat(x - outMinX) + 0.5) / CGFloat(outCropWidth)
                let sourceX = u * CGFloat(cropWidth) - 0.5
                fullAlpha[y * outputSize + x] = bilinearAlpha(cropAlpha, width: cropWidth, height: cropHeight, x: sourceX, y: sourceY)
            }
        }

        return UVLipAlphaMask(
            width: outputSize,
            height: outputSize,
            alpha: fullAlpha
        )
    }

    private func featheredFullUVAlpha(_ alpha: [UInt8], width: Int, height: Int) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else { return alpha }

        let softened = blurred(
            alpha: blurred(alpha: alpha, width: width, height: height, radius: 1),
            width: width,
            height: height,
            radius: 1
        )
        var output = alpha
        for index in alpha.indices {
            let original = CGFloat(alpha[index])
            let soft = CGFloat(softened[index])
            if original > 36 {
                output[index] = UInt8(min(max(max(original * 0.90, soft * 1.04), 0), 238))
            } else {
                output[index] = UInt8(min(max(soft * 0.70, 0), 150))
            }
        }

        return output
    }

    private static func signedArea(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        ((p1.x - p0.x) * (p2.y - p0.y)) - ((p1.y - p0.y) * (p2.x - p0.x))
    }

    private static func lipColorFeature(_ color: ARKitPixelColor) -> CGFloat {
        let redGreen = color.red - color.green
        let redBlue = color.red - color.blue
        return (redGreen * 0.62) + (redBlue * 0.22) + (color.saturation * 0.16)
    }

    private static func percentile(_ values: [CGFloat], fraction: CGFloat) -> CGFloat {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = min(max(Int(round(min(max(fraction, 0), 1) * CGFloat(sorted.count - 1))), 0), sorted.count - 1)
        return sorted[index]
    }

    private static func smoothstep(_ value: CGFloat) -> CGFloat {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }

    @inline(__always)
    private func smoothstep(_ value: Float) -> Float {
        let clamped = min(max(value, 0), 1)
        return clamped * clamped * (3 - 2 * clamped)
    }

    @inline(__always)
    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private func bilinearAlpha(_ alpha: [UInt8], width: Int, height: Int, x: CGFloat, y: CGFloat) -> UInt8 {
        let clampedX = min(max(x, 0), CGFloat(width - 1))
        let clampedY = min(max(y, 0), CGFloat(height - 1))
        let x0 = min(max(Int(floor(clampedX)), 0), width - 1)
        let y0 = min(max(Int(floor(clampedY)), 0), height - 1)
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = clampedX - CGFloat(x0)
        let ty = clampedY - CGFloat(y0)
        let a00 = CGFloat(alpha[y0 * width + x0])
        let a10 = CGFloat(alpha[y0 * width + x1])
        let a01 = CGFloat(alpha[y1 * width + x0])
        let a11 = CGFloat(alpha[y1 * width + x1])
        let top = a00 + (a10 - a00) * tx
        let bottom = a01 + (a11 - a01) * tx
        return UInt8(min(max(top + (bottom - top) * ty, 0), 255))
    }

    private func removeTinyComponents(alpha: [UInt8], width: Int, height: Int, threshold: UInt8) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else { return alpha }

        var output = alpha
        var visited = [Bool](repeating: false, count: alpha.count)
        var stack: [Int] = []
        var component: [Int] = []
        let minimumPixels = max(12, Int(CGFloat(width * height) * 0.0006))

        for start in alpha.indices where !visited[start] && alpha[start] > threshold {
            stack.removeAll(keepingCapacity: true)
            component.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true

            while let index = stack.popLast() {
                component.append(index)
                let x = index % width
                let y = index / width

                if x > 0 {
                    let neighbor = index - 1
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold {
                            stack.append(neighbor)
                        }
                    }
                }
                if x < width - 1 {
                    let neighbor = index + 1
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold {
                            stack.append(neighbor)
                        }
                    }
                }
                if y > 0 {
                    let neighbor = index - width
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold {
                            stack.append(neighbor)
                        }
                    }
                }
                if y < height - 1 {
                    let neighbor = index + width
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        if alpha[neighbor] > threshold {
                            stack.append(neighbor)
                        }
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

    private func expanded(alpha: [UInt8], width: Int, height: Int, threshold: UInt8) -> [UInt8] {
        guard width > 1, height > 1, alpha.count == width * height else { return alpha }

        var output = alpha
        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width where alpha[rowOffset + x] <= threshold {
                var best = UInt8(0)
                let minY = max(0, y - 1)
                let maxY = min(height - 1, y + 1)
                let minX = max(0, x - 1)
                let maxX = min(width - 1, x + 1)
                for sampleY in minY...maxY {
                    let sampleRowOffset = sampleY * width
                    for sampleX in minX...maxX {
                        best = max(best, alpha[sampleRowOffset + sampleX])
                    }
                }
                if best > threshold {
                    output[rowOffset + x] = UInt8(CGFloat(best) * 0.42)
                }
            }
        }

        return output
    }

    private func softClipped(alpha: [UInt8]) -> [UInt8] {
        alpha.map { value in
            guard value > 22 else { return 0 }
            let normalized = min(max((CGFloat(value) - 22) / 190, 0), 1)
            let clipped = pow(normalized, 0.92)
            return UInt8(min(max(clipped * 238, 0), 238))
        }
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

struct ARKitPixelColor {
    let red: CGFloat
    let green: CGFloat
    let blue: CGFloat

    static let black = ARKitPixelColor(red: 0, green: 0, blue: 0)

    var saturation: CGFloat {
        let maxValue = max(red, green, blue)
        let minValue = min(red, green, blue)
        return maxValue == 0 ? 0 : (maxValue - minValue) / maxValue
    }
}

struct ARKitPixelSampler {
    private enum Storage {
        case bgra(base: UnsafeMutablePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int)
        case ycbcr(
            yBase: UnsafeMutablePointer<UInt8>,
            cbcrBase: UnsafeMutablePointer<UInt8>,
            yWidth: Int,
            yHeight: Int,
            yBytesPerRow: Int,
            cbcrWidth: Int,
            cbcrHeight: Int,
            cbcrBytesPerRow: Int
        )
    }

    private let storage: Storage

    init?(pixelBuffer: CVPixelBuffer) {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        if
            format == kCVPixelFormatType_32BGRA,
            let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        {
            storage = .bgra(
                base: baseAddress.assumingMemoryBound(to: UInt8.self),
                width: CVPixelBufferGetWidth(pixelBuffer),
                height: CVPixelBufferGetHeight(pixelBuffer),
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer)
            )
            return
        }

        if
            (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            CVPixelBufferGetPlaneCount(pixelBuffer) >= 2,
            let yBaseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
            let cbcrBaseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1)
        {
            storage = .ycbcr(
                yBase: yBaseAddress.assumingMemoryBound(to: UInt8.self),
                cbcrBase: cbcrBaseAddress.assumingMemoryBound(to: UInt8.self),
                yWidth: CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
                yHeight: CVPixelBufferGetHeightOfPlane(pixelBuffer, 0),
                yBytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0),
                cbcrWidth: CVPixelBufferGetWidthOfPlane(pixelBuffer, 1),
                cbcrHeight: CVPixelBufferGetHeightOfPlane(pixelBuffer, 1),
                cbcrBytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
            )
            return
        }

        return nil
    }

    func fillNormalizedRGB(
        into pointer: UnsafeMutablePointer<Float>,
        strides: [Int],
        inputWidth: Int,
        inputHeight: Int,
        topLeft: CGPoint,
        topRight: CGPoint,
        bottomLeft: CGPoint,
        bottomRight: CGPoint,
        mean: [Float],
        std: [Float]
    ) {
        guard inputWidth > 0, inputHeight > 0, strides.count == 4, mean.count >= 3, std.count >= 3 else {
            return
        }

        let channelStride = strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]
        let redBase = 0
        let greenBase = channelStride
        let blueBase = channelStride * 2
        let redMean = mean[0]
        let greenMean = mean[1]
        let blueMean = mean[2]
        let redInvStd = 1 / std[0]
        let greenInvStd = 1 / std[1]
        let blueInvStd = 1 / std[2]

        let tlx = Float(topLeft.x)
        let tly = Float(topLeft.y)
        let trx = Float(topRight.x)
        let tryValue = Float(topRight.y)
        let blx = Float(bottomLeft.x)
        let bly = Float(bottomLeft.y)
        let brx = Float(bottomRight.x)
        let bry = Float(bottomRight.y)
        let invInputWidth = 1 / Float(inputWidth)
        let invInputHeight = 1 / Float(inputHeight)

        switch storage {
        case .bgra(let base, let width, let height, let bytesPerRow):
            let maxX = Float(max(width - 1, 0))
            let maxY = Float(max(height - 1, 0))

            for y in 0..<inputHeight {
                let v = (Float(y) + 0.5) * invInputHeight
                let leftX = tlx + ((blx - tlx) * v)
                let leftY = tly + ((bly - tly) * v)
                let rightX = trx + ((brx - trx) * v)
                let rightY = tryValue + ((bry - tryValue) * v)
                let rowBase = y * rowStride

                for x in 0..<inputWidth {
                    let u = (Float(x) + 0.5) * invInputWidth
                    let imageX = leftX + ((rightX - leftX) * u)
                    let imageY = leftY + ((rightY - leftY) * u)
                    let pixelX = clampedRoundedIndex(imageX * maxX, upperBound: width)
                    let pixelY = clampedRoundedIndex(imageY * maxY, upperBound: height)
                    let pixelOffset = pixelY * bytesPerRow + pixelX * 4
                    let outputOffset = rowBase + x * columnStride
                    let red = Float(base[pixelOffset + 2]) / 255
                    let green = Float(base[pixelOffset + 1]) / 255
                    let blue = Float(base[pixelOffset]) / 255

                    pointer[redBase + outputOffset] = (red - redMean) * redInvStd
                    pointer[greenBase + outputOffset] = (green - greenMean) * greenInvStd
                    pointer[blueBase + outputOffset] = (blue - blueMean) * blueInvStd
                }
            }

        case .ycbcr(
            let yBase,
            let cbcrBase,
            let yWidth,
            let yHeight,
            let yBytesPerRow,
            let cbcrWidth,
            let cbcrHeight,
            let cbcrBytesPerRow
        ):
            let maxYPlaneX = Float(max(yWidth - 1, 0))
            let maxYPlaneY = Float(max(yHeight - 1, 0))
            let maxChromaX = max(cbcrWidth - 1, 0)
            let maxChromaY = max(cbcrHeight - 1, 0)

            for y in 0..<inputHeight {
                let v = (Float(y) + 0.5) * invInputHeight
                let leftX = tlx + ((blx - tlx) * v)
                let leftY = tly + ((bly - tly) * v)
                let rightX = trx + ((brx - trx) * v)
                let rightY = tryValue + ((bry - tryValue) * v)
                let rowBase = y * rowStride

                for x in 0..<inputWidth {
                    let u = (Float(x) + 0.5) * invInputWidth
                    let imageX = leftX + ((rightX - leftX) * u)
                    let imageY = leftY + ((rightY - leftY) * u)
                    let lumaX = clampedRoundedIndex(imageX * maxYPlaneX, upperBound: yWidth)
                    let lumaY = clampedRoundedIndex(imageY * maxYPlaneY, upperBound: yHeight)
                    let chromaX = min(max(lumaX * maxChromaX / max(yWidth - 1, 1), 0), maxChromaX)
                    let chromaY = min(max(lumaY * maxChromaY / max(yHeight - 1, 1), 0), maxChromaY)

                    let yValue = Float(yBase[lumaY * yBytesPerRow + lumaX]) / 255
                    let chromaOffset = chromaY * cbcrBytesPerRow + chromaX * 2
                    let cb = Float(cbcrBase[chromaOffset]) / 255 - 0.5
                    let cr = Float(cbcrBase[chromaOffset + 1]) / 255 - 0.5
                    let red = min(max(yValue + 1.402 * cr, 0), 1)
                    let green = min(max(yValue - 0.344_136 * cb - 0.714_136 * cr, 0), 1)
                    let blue = min(max(yValue + 1.772 * cb, 0), 1)
                    let outputOffset = rowBase + x * columnStride

                    pointer[redBase + outputOffset] = (red - redMean) * redInvStd
                    pointer[greenBase + outputOffset] = (green - greenMean) * greenInvStd
                    pointer[blueBase + outputOffset] = (blue - blueMean) * blueInvStd
                }
            }
        }
    }

    func sample(normalizedX: CGFloat, normalizedY: CGFloat) -> ARKitPixelColor? {
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

        switch storage {
        case .bgra(let base, let width, let height, let bytesPerRow):
            let x = min(max(Int((normalizedX * CGFloat(width - 1)).rounded()), 0), width - 1)
            let y = min(max(Int((normalizedY * CGFloat(height - 1)).rounded()), 0), height - 1)
            let offset = y * bytesPerRow + x * 4

            return ARKitPixelColor(
                red: CGFloat(base[offset + 2]) / 255,
                green: CGFloat(base[offset + 1]) / 255,
                blue: CGFloat(base[offset]) / 255
            )

        case .ycbcr(
            let yBase,
            let cbcrBase,
            let yWidth,
            let yHeight,
            let yBytesPerRow,
            let cbcrWidth,
            let cbcrHeight,
            let cbcrBytesPerRow
        ):
            let x = min(max(Int((normalizedX * CGFloat(yWidth - 1)).rounded()), 0), yWidth - 1)
            let y = min(max(Int((normalizedY * CGFloat(yHeight - 1)).rounded()), 0), yHeight - 1)
            let chromaX = min(max(Int((CGFloat(x) / CGFloat(max(yWidth - 1, 1))) * CGFloat(cbcrWidth - 1)), 0), cbcrWidth - 1)
            let chromaY = min(max(Int((CGFloat(y) / CGFloat(max(yHeight - 1, 1))) * CGFloat(cbcrHeight - 1)), 0), cbcrHeight - 1)
            let yValue = CGFloat(yBase[y * yBytesPerRow + x]) / 255
            let chromaOffset = chromaY * cbcrBytesPerRow + chromaX * 2
            let cb = CGFloat(cbcrBase[chromaOffset]) / 255 - 0.5
            let cr = CGFloat(cbcrBase[chromaOffset + 1]) / 255 - 0.5

            return ARKitPixelColor(
                red: min(max(yValue + 1.402 * cr, 0), 1),
                green: min(max(yValue - 0.344_136 * cb - 0.714_136 * cr, 0), 1),
                blue: min(max(yValue + 1.772 * cb, 0), 1)
            )
        }
    }

    @inline(__always)
    private func clampedRoundedIndex(_ value: Float, upperBound: Int) -> Int {
        guard upperBound > 1 else { return 0 }
        let rounded = Int(value + 0.5)
        if rounded < 0 { return 0 }
        if rounded >= upperBound { return upperBound - 1 }
        return rounded
    }
}

private extension CGPoint {
    func interpolated(to point: CGPoint, amount: CGFloat) -> CGPoint {
        CGPoint(
            x: x + (point.x - x) * amount,
            y: y + (point.y - y) * amount
        )
    }
}
