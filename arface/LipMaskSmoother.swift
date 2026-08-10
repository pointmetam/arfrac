import Foundation
import CoreGraphics
import Vision

@available(iOS 14.0, *)
final class LipMaskSmoother {
    
    static func smoothMask(width: Int, height: Int, alpha: [UInt8], threshold: UInt8 = 127) -> [UInt8]? {
        // 1. Threshold and create CGImage
        var binarized = [UInt8](repeating: 0, count: width * height)
        for i in 0..<alpha.count {
            binarized[i] = alpha[i] >= threshold ? 255 : 0
        }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(data: &binarized,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let cgImage = context.makeImage() else {
            return nil
        }
        
        // 2. Vision Request
        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = false
        request.maximumImageDimension = max(width, height)
        
        let handler = VNImageRequestHandler(cgImage: cgImage)
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        
        guard let observation = request.results?.first as? VNContoursObservation else { return nil }
        
        // Find largest top-level contour
        var largestContour: VNContour?
        var maxPoints = 0
        for contour in observation.topLevelContours {
            if contour.pointCount > maxPoints {
                maxPoints = contour.pointCount
                largestContour = contour
            }
        }
        
        guard let outer = largestContour else { return nil }
        
        // 3. Extract and smooth points
        let outerPoints = extractAndSmooth(contour: outer, width: width, height: height, targetPoints: 96)
        
        var holes = [[CGPoint]]()
        for child in outer.childContours {
            if child.pointCount > 10 {
                let holePoints = extractAndSmooth(contour: child, width: width, height: height, targetPoints: 48)
                holes.append(holePoints)
            }
        }
        
        // 4. Draw to new mask
        var resultAlpha = [UInt8](repeating: 0, count: width * height)
        guard let outContext = CGContext(data: &resultAlpha,
                                         width: width,
                                         height: height,
                                         bitsPerComponent: 8,
                                         bytesPerRow: width,
                                         space: colorSpace,
                                         bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return nil
        }
        
        // Disable antialiasing if we want a hard mask, or enable for soft edges
        outContext.setShouldAntialias(true)
        outContext.setFillColor(gray: 1.0, alpha: 1.0)
        
        drawPolygon(points: outerPoints, to: outContext)
        outContext.fillPath()
        
        outContext.setFillColor(gray: 0.0, alpha: 1.0)
        outContext.setBlendMode(.clear)
        for hole in holes {
            drawPolygon(points: hole, to: outContext)
            outContext.fillPath()
        }
        
        return resultAlpha
    }
    
    private static func drawPolygon(points: [CGPoint], to ctx: CGContext) {
        if points.isEmpty { return }
        ctx.move(to: points[0])
        for p in points.dropFirst() {
            ctx.addLine(to: p)
        }
        ctx.closePath()
    }
    
    private static func extractAndSmooth(contour: VNContour, width: Int, height: Int, targetPoints: Int) -> [CGPoint] {
        let normalized = contour.normalizedPoints
        let points = normalized.map { CGPoint(x: CGFloat($0.x) * CGFloat(width), y: CGFloat(1.0 - $0.y) * CGFloat(height)) }
        
        // 1. Calculate cumulative distances
        var distances = [CGFloat]()
        distances.append(0)
        var totalLength: CGFloat = 0
        for i in 0..<points.count {
            let p1 = points[i]
            let p2 = points[(i + 1) % points.count]
            let dx = p2.x - p1.x
            let dy = p2.y - p1.y
            let dist = sqrt(dx*dx + dy*dy)
            totalLength += dist
            distances.append(totalLength)
        }
        
        if totalLength <= 1.0 {
            return points
        }
        
        // 2. Resample evenly
        var resampled = [CGPoint]()
        var currentDist: CGFloat = 0
        let step = totalLength / CGFloat(targetPoints)
        
        var idx = 0
        for _ in 0..<targetPoints {
            while idx < points.count && distances[idx + 1] < currentDist {
                idx += 1
            }
            if idx >= points.count { idx = points.count - 1 }
            
            let p1 = points[idx]
            let p2 = points[(idx + 1) % points.count]
            let d1 = distances[idx]
            let d2 = distances[idx + 1]
            
            let t = (d2 - d1) > 0.001 ? (currentDist - d1) / (d2 - d1) : 0
            let nx = p1.x + (p2.x - p1.x) * t
            let ny = p1.y + (p2.y - p1.y) * t
            resampled.append(CGPoint(x: nx, y: ny))
            
            currentDist += step
        }
        
        // 3. Catmull-Rom smoothing
        return catmullRom(points: resampled, steps: 4)
    }
    
    private static func catmullRom(points: [CGPoint], steps: Int) -> [CGPoint] {
        guard points.count >= 3 else { return points }
        var result = [CGPoint]()
        
        let n = points.count
        for i in 0..<n {
            let p0 = points[(i - 1 + n) % n]
            let p1 = points[i]
            let p2 = points[(i + 1) % n]
            let p3 = points[(i + 2) % n]
            
            for j in 0..<steps {
                let t = CGFloat(j) / CGFloat(steps)
                let t2 = t * t
                let t3 = t2 * t
                
                let f0 = -0.5 * t3 + t2 - 0.5 * t
                let f1 = 1.5 * t3 - 2.5 * t2 + 1.0
                let f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                let f3 = 0.5 * t3 - 0.5 * t2
                
                let x = p0.x * f0 + p1.x * f1 + p2.x * f2 + p3.x * f3
                let y = p0.y * f0 + p1.y * f1 + p2.y * f2 + p3.y * f3
                result.append(CGPoint(x: x, y: y))
            }
        }
        return result
    }
}
