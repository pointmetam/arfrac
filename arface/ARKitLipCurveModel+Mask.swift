import Foundation
import CoreGraphics
import QuartzCore

extension ARKitLipCurveModel {
    struct TimedCurveMask {
        let mask: ARKitLipMask?
        let uvMask: UVLipAlphaMask?
        let info: String
    }
    
    func timedMask(
        request: ARKitLipParsingRequest,
        projectionMap: ARKitFaceUVProjectionMap? = nil
    ) -> TimedCurveMask {
        let start = CACurrentMediaTime()
        guard let curves = predict(request: request) else {
            return TimedCurveMask(mask: nil, uvMask: nil, info: "predict failed")
        }
        
        let width = self.inputWidth
        let height = self.inputHeight
        var alpha = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        guard curves.count == 4, let context = CGContext(data: &alpha, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return TimedCurveMask(mask: nil, uvMask: nil, info: "cgcontext failed")
        }
        
        context.setShouldAntialias(true)
        context.setFillColor(gray: 1.0, alpha: 1.0)
        
        let maxX = CGFloat(max(width - 1, 1))
        let maxY = CGFloat(max(height - 1, 1))
        let mapPoint: (CGPoint) -> CGPoint = { p in
            CGPoint(
                x: min(max(p.x, 0), 1) * maxX,
                y: min(max(p.y, 0), 1) * maxY
            )
        }
        
        let orderedCurves = curves.map(normalizedCurveDirection)
        // Keep the boundary smooth, but clamp interpolation so corners do not overshoot.
        let smoothedCurves = orderedCurves.map { catmullRomOpen(points: $0, steps: 6) }
        let screenCurves = smoothedCurves.map { curve in
            curve.map { point in
                CGPoint(
                    x: request.roi.minX + min(max(point.x, 0), 1) * request.roi.width,
                    y: request.roi.minY + min(max(point.y, 0), 1) * request.roi.height
                )
            }
        }

        // Match the training/export semantic order:
        // 0 upper_outer, 1 upper_inner, 2 lower_inner, 3 lower_outer.
        var upperLipPoints = smoothedCurves[0].map(mapPoint)
        upperLipPoints.append(contentsOf: smoothedCurves[1].reversed().map(mapPoint))
        drawPolygon(points: upperLipPoints, in: context)
        context.fillPath()

        var lowerLipPoints = smoothedCurves[3].map(mapPoint)
        lowerLipPoints.append(contentsOf: smoothedCurves[2].reversed().map(mapPoint))
        context.setBlendMode(.normal)
        drawPolygon(points: lowerLipPoints, in: context)
        context.fillPath()
        
        let ms = (CACurrentMediaTime() - start) * 1000
        
        // Use the screen-space ROI so downstream projection sampling matches the same coordinate space.
        let mask = ARKitLipMask(frame: request.roi, width: width, height: height, alpha: alpha)
        let uvMask = projectionMap?.uvLipAlphaMask(fromScreenCurves: screenCurves)
        return TimedCurveMask(
            mask: mask,
            uvMask: uvMask,
            info: String(format: "curve %.1fms fixed 0/1/2/3", ms)
        )
    }
    
    private func drawPolygon(points: [CGPoint], in context: CGContext) {
        if points.isEmpty { return }
        context.move(to: points[0])
        for p in points.dropFirst() {
            context.addLine(to: p)
        }
        context.closePath()
    }
    
    private func catmullRomOpen(points: [CGPoint], steps: Int) -> [CGPoint] {
        guard points.count >= 2 else { return points }
        var result = [CGPoint]()
        result.reserveCapacity(points.count * steps)
        
        // Duplicate endpoints instead of reflecting them, which reduces corner overshoot.
        var extPoints = [CGPoint]()
        extPoints.reserveCapacity(points.count + 2)
        extPoints.append(points[0])
        extPoints.append(contentsOf: points)
        extPoints.append(points[points.count - 1])
        
        let n = extPoints.count
        for i in 1..<(n - 2) {
            let p0 = extPoints[i - 1]
            let p1 = extPoints[i]
            let p2 = extPoints[i + 1]
            let p3 = extPoints[i + 2]
            let minX = min(p1.x, p2.x)
            let maxX = max(p1.x, p2.x)
            let minY = min(p1.y, p2.y)
            let maxY = max(p1.y, p2.y)
            
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
                result.append(
                    CGPoint(
                        x: min(max(min(max(x, minX), maxX), 0), 1),
                        y: min(max(min(max(y, minY), maxY), 0), 1)
                    )
                )
            }
        }
        
        // Add the very last point
        if let lastPoint = points.last {
            result.append(
                CGPoint(
                    x: min(max(lastPoint.x, 0), 1),
                    y: min(max(lastPoint.y, 0), 1)
                )
            )
        }
        
        return result
    }

    private func normalizedCurveDirection(points: [CGPoint]) -> [CGPoint] {
        guard let first = points.first, let last = points.last else { return points }
        return first.x <= last.x ? points : points.reversed()
    }
}
