//
//  LipMaskFaceMeshProjector.swift
//  arface
//
//  Created by Codex on 2026/5/27.
//

import CoreGraphics
import QuartzCore
import SceneKit
import simd

struct LipMaskFaceMeshProjectionVertex: Sendable {
    let position: SIMD3<Float>
    let maskUV: SIMD2<Float>
}

struct LipMaskFaceMeshProjectionResult: @unchecked Sendable {
    let geometry: SCNGeometry
    let durationMS: Double
}

final class LipMaskFaceMeshProjector: @unchecked Sendable {
    private let triangleSubdivision = 5

    func makeGeometry(
        mask: ARKitLipMask,
        vertices: [LipMaskFaceMeshProjectionVertex],
        lipColor: SIMD4<Float>
    ) -> LipMaskFaceMeshProjectionResult? {
        guard
            mask.width > 1,
            mask.height > 1,
            mask.alpha.count == mask.width * mask.height,
            vertices.count >= 3,
            vertices.count < Int(UInt16.max)
        else {
            return nil
        }

        let start = CACurrentMediaTime()
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        var indices: [UInt16] = []
        let estimatedVertexCount = min(vertices.count * triangleSubdivision * triangleSubdivision, Int(UInt16.max) - 1)
        positions.reserveCapacity(estimatedVertexCount)
        colors.reserveCapacity(estimatedVertexCount)
        indices.reserveCapacity(estimatedVertexCount)

        for triangleStart in stride(from: 0, to: vertices.count - 2, by: 3) {
            appendSubdividedTriangle(
                vertices[triangleStart],
                vertices[triangleStart + 1],
                vertices[triangleStart + 2],
                mask: mask,
                lipColor: lipColor,
                positions: &positions,
                colors: &colors,
                indices: &indices
            )
        }

        guard positions.count >= 9, colors.count == positions.count, indices.count == positions.count else {
            return nil
        }

        let positionData = positions.withUnsafeBufferPointer { Data(buffer: $0) }
        let colorData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0) }

        let positionSource = SCNGeometrySource(
            data: positionData,
            semantic: .vertex,
            vectorCount: positions.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride
        )
        let colorSource = SCNGeometrySource(
            data: colorData,
            semantic: .color,
            vectorCount: colors.count,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride
        )
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<UInt16>.size
        )
        let geometry = SCNGeometry(sources: [positionSource, colorSource], elements: [element])

        return LipMaskFaceMeshProjectionResult(
            geometry: geometry,
            durationMS: (CACurrentMediaTime() - start) * 1000
        )
    }

    private func appendSubdividedTriangle(
        _ v0: LipMaskFaceMeshProjectionVertex,
        _ v1: LipMaskFaceMeshProjectionVertex,
        _ v2: LipMaskFaceMeshProjectionVertex,
        mask: ARKitLipMask,
        lipColor: SIMD4<Float>,
        positions: inout [SIMD3<Float>],
        colors: inout [SIMD4<Float>],
        indices: inout [UInt16]
    ) {
        let subdivision = max(triangleSubdivision, 1)
        for row in 0..<subdivision {
            for column in 0..<(subdivision - row) {
                let u0 = Float(column) / Float(subdivision)
                let v0Progress = Float(row) / Float(subdivision)
                let u1 = Float(column + 1) / Float(subdivision)
                let v1Progress = Float(row) / Float(subdivision)
                let u2 = Float(column) / Float(subdivision)
                let v2Progress = Float(row + 1) / Float(subdivision)

                appendTriangle(
                    interpolatedVertex(v0, v1, v2, u: u0, v: v0Progress),
                    interpolatedVertex(v0, v1, v2, u: u1, v: v1Progress),
                    interpolatedVertex(v0, v1, v2, u: u2, v: v2Progress),
                    mask: mask,
                    lipColor: lipColor,
                    positions: &positions,
                    colors: &colors,
                    indices: &indices
                )

                guard row + column < subdivision - 1 else { continue }
                let u3 = Float(column + 1) / Float(subdivision)
                let v3Progress = Float(row + 1) / Float(subdivision)
                appendTriangle(
                    interpolatedVertex(v0, v1, v2, u: u1, v: v1Progress),
                    interpolatedVertex(v0, v1, v2, u: u3, v: v3Progress),
                    interpolatedVertex(v0, v1, v2, u: u2, v: v2Progress),
                    mask: mask,
                    lipColor: lipColor,
                    positions: &positions,
                    colors: &colors,
                    indices: &indices
                )
            }
        }
    }

    private func appendTriangle(
        _ v0: LipMaskFaceMeshProjectionVertex,
        _ v1: LipMaskFaceMeshProjectionVertex,
        _ v2: LipMaskFaceMeshProjectionVertex,
        mask: ARKitLipMask,
        lipColor: SIMD4<Float>,
        positions: inout [SIMD3<Float>],
        colors: inout [SIMD4<Float>],
        indices: inout [UInt16]
    ) {
        let a0 = sampledAlpha(mask: mask, uv: v0.maskUV)
        let a1 = sampledAlpha(mask: mask, uv: v1.maskUV)
        let a2 = sampledAlpha(mask: mask, uv: v2.maskUV)
        guard max(a0, a1, a2) > 0.015, positions.count + 3 < Int(UInt16.max) else { return }

        append(v0, alpha: a0, lipColor: lipColor, positions: &positions, colors: &colors, indices: &indices)
        append(v1, alpha: a1, lipColor: lipColor, positions: &positions, colors: &colors, indices: &indices)
        append(v2, alpha: a2, lipColor: lipColor, positions: &positions, colors: &colors, indices: &indices)
    }

    private func interpolatedVertex(
        _ v0: LipMaskFaceMeshProjectionVertex,
        _ v1: LipMaskFaceMeshProjectionVertex,
        _ v2: LipMaskFaceMeshProjectionVertex,
        u: Float,
        v: Float
    ) -> LipMaskFaceMeshProjectionVertex {
        let w = 1 - u - v
        return LipMaskFaceMeshProjectionVertex(
            position: (v0.position * w) + (v1.position * u) + (v2.position * v),
            maskUV: (v0.maskUV * w) + (v1.maskUV * u) + (v2.maskUV * v)
        )
    }

    private func append(
        _ vertex: LipMaskFaceMeshProjectionVertex,
        alpha: Float,
        lipColor: SIMD4<Float>,
        positions: inout [SIMD3<Float>],
        colors: inout [SIMD4<Float>],
        indices: inout [UInt16]
    ) {
        let sharpenedAlpha = edgeSharpenedAlpha(alpha) * lipColor.w
        positions.append(vertex.position * 1.002)
        colors.append(SIMD4<Float>(lipColor.x, lipColor.y, lipColor.z, min(max(sharpenedAlpha, 0), 1)))
        indices.append(UInt16(positions.count - 1))
    }

    private func sampledAlpha(mask: ARKitLipMask, uv: SIMD2<Float>) -> Float {
        guard uv.x.isFinite, uv.y.isFinite, uv.x >= 0, uv.x <= 1, uv.y >= 0, uv.y <= 1 else {
            return 0
        }

        let x = uv.x * Float(mask.width - 1)
        let y = uv.y * Float(mask.height - 1)
        let x0 = min(max(Int(floor(x)), 0), mask.width - 1)
        let y0 = min(max(Int(floor(y)), 0), mask.height - 1)
        let x1 = min(x0 + 1, mask.width - 1)
        let y1 = min(y0 + 1, mask.height - 1)
        let tx = x - Float(x0)
        let ty = y - Float(y0)

        let a00 = Float(mask.alpha[y0 * mask.width + x0]) / 255
        let a10 = Float(mask.alpha[y0 * mask.width + x1]) / 255
        let a01 = Float(mask.alpha[y1 * mask.width + x0]) / 255
        let a11 = Float(mask.alpha[y1 * mask.width + x1]) / 255
        let top = a00 + (a10 - a00) * tx
        let bottom = a01 + (a11 - a01) * tx
        return top + (bottom - top) * ty
    }

    private func edgeSharpenedAlpha(_ alpha: Float) -> Float {
        let low: Float = 0.08
        let high: Float = 0.78
        let normalized = min(max((alpha - low) / (high - low), 0), 1)
        let smoothed = normalized * normalized * (3 - 2 * normalized)
        return min(max(smoothed * 1.04, 0), 1)
    }
}
