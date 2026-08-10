//
//  FaceParsingMetalCompositor.swift
//  arface
//
//  Created by lee on 2026/5/25.
//

import CoreGraphics
import Foundation
import Metal

final class FaceParsingMetalCompositor {
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let colorizePipeline: MTLComputePipelineState?
    private let debugPaletteBuffer: MTLBuffer?

    init() {
        guard
            let device = MTLCreateSystemDefaultDevice(),
            let commandQueue = device.makeCommandQueue(),
            let library = try? device.makeLibrary(source: Self.shaderSource, options: nil),
            let colorizeFunction = library.makeFunction(name: "faceParsingDebugColorize")
        else {
            self.device = nil
            self.commandQueue = nil
            self.colorizePipeline = nil
            self.debugPaletteBuffer = nil
            return
        }

        self.device = device
        self.commandQueue = commandQueue
        self.colorizePipeline = try? device.makeComputePipelineState(function: colorizeFunction)
        self.debugPaletteBuffer = ARKitFaceParsingMask.debugPalette.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: baseAddress,
                length: bytes.count,
                options: [.storageModeShared]
            )
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void faceParsingDebugColorize(
        const device uchar *classIds [[buffer(0)]],
        const device uchar *confidence [[buffer(1)]],
        const device uchar4 *palette [[buffer(2)]],
        device uchar4 *output [[buffer(3)]],
        constant uint &pixelCount [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= pixelCount) {
            return;
        }

        const uchar classId = classIds[id];
        const uchar conf = confidence[id];
        if (classId == 0 || conf <= 12) {
            output[id] = uchar4(0, 0, 0, 0);
            return;
        }

        const uint paletteIndex = min((uint)classId, 8u);
        const uchar4 color = palette[paletteIndex];
        const ushort alpha = (ushort(color.a) * ushort(conf)) / 255u;

        output[id] = uchar4(
            uchar((ushort(color.r) * alpha) / 255u),
            uchar((ushort(color.g) * alpha) / 255u),
            uchar((ushort(color.b) * alpha) / 255u),
            uchar(alpha)
        );
    }
    """

    func makeDebugImage(from mask: ARKitFaceParsingMask) -> CGImage? {
        guard
            let device,
            let commandQueue,
            let colorizePipeline,
            let debugPaletteBuffer,
            mask.width > 0,
            mask.height > 0,
            mask.classIds.count == mask.width * mask.height,
            mask.confidence.count == mask.width * mask.height
        else {
            return nil
        }

        let pixelCount = mask.width * mask.height
        let outputLength = pixelCount * 4
        let classBuffer = mask.classIds.withUnsafeBytes { bytes -> MTLBuffer? in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: bytes.count, options: [.storageModeShared])
        }
        let confidenceBuffer = mask.confidence.withUnsafeBytes { bytes -> MTLBuffer? in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: bytes.count, options: [.storageModeShared])
        }

        guard
            let classBuffer,
            let confidenceBuffer,
            let outputBuffer = device.makeBuffer(length: outputLength, options: [.storageModeShared]),
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }

        var count = UInt32(pixelCount)
        encoder.setComputePipelineState(colorizePipeline)
        encoder.setBuffer(classBuffer, offset: 0, index: 0)
        encoder.setBuffer(confidenceBuffer, offset: 0, index: 1)
        encoder.setBuffer(debugPaletteBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)

        let threadWidth = min(colorizePipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (pixelCount + threadWidth - 1) / threadWidth,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        guard commandBuffer.status == .completed else { return nil }

        let data = Data(
            bytes: outputBuffer.contents().assumingMemoryBound(to: UInt8.self),
            count: outputLength
        )
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        return CGImage(
            width: mask.width,
            height: mask.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: mask.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}
