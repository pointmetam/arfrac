import Foundation
import CoreGraphics
import CoreML
import Accelerate
import ARKit

final class ARKitLipCurveModel: @unchecked Sendable {
    private static let defaultInputWidth = 384
    private static let defaultInputHeight = 256
    private static let defaultInputChannels = 3
    
    private let model: MLModel?
    private let inputName: String?
    private let outputName: String?
    let inputWidth: Int
    let inputHeight: Int
    let inputChannels: Int
    private let resourceName: String
    private let curveCount: Int
    private let pointsPerCurve: Int

    init(resourceName: String = "LipCurveStudent384") {
        self.resourceName = resourceName
        
        do {
            let url = try Self.loadableModelURL(resourceName: resourceName)
            let config = MLModelConfiguration()
            config.computeUnits = .all
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            self.model = nil
        }
        
        let inputDescription = self.model?.modelDescription.inputDescriptionsByName.first?.value
        let outputDescription = self.model?.modelDescription.outputDescriptionsByName.first?.value
        
        self.inputName = inputDescription?.name
        self.outputName = outputDescription?.name
        
        if let shape = inputDescription?.multiArrayConstraint?.shape, shape.count == 4 {
            self.inputChannels = shape[1].intValue
            self.inputHeight = shape[2].intValue
            self.inputWidth = shape[3].intValue
        } else {
            self.inputChannels = Self.defaultInputChannels
            self.inputHeight = Self.defaultInputHeight
            self.inputWidth = Self.defaultInputWidth
        }
        
        if let outShape = outputDescription?.multiArrayConstraint?.shape, outShape.count == 4 {
            self.curveCount = outShape[1].intValue
            self.pointsPerCurve = outShape[2].intValue
        } else {
            self.curveCount = 4
            self.pointsPerCurve = 24
        }
    }
    
    var isReady: Bool { model != nil }
    
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
    
    private static func imagePoint(from viewPoint: CGPoint, viewportSize: CGSize, transform: CGAffineTransform) -> CGPoint {
        let normalized = CGPoint(x: viewPoint.x / viewportSize.width, y: viewPoint.y / viewportSize.height)
        let transformed = normalized.applying(transform)
        return CGPoint(x: transformed.x, y: transformed.y) // Normalized [0, 1]
    }
    
    func predict(request: ARKitLipParsingRequest) -> [[CGPoint]]? {
        guard let model, let inputName, let outputName else { return nil }
        
        do {
            let inputArray = try makeInputArray(request: request)
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
            let result = try model.prediction(from: provider)
            guard let outArray = result.featureValue(for: outputName)?.multiArrayValue else { return nil }
            
            let shape = outArray.shape.map { $0.intValue }
            guard shape.count == 4, shape[0] == 1, shape[1] == curveCount, shape[2] == pointsPerCurve, shape[3] == 2 else { return nil }
            
            var curves = [[CGPoint]]()
            let ptr = outArray.dataPointer.bindMemory(to: Float32.self, capacity: outArray.count)
            var offset = 0
            for _ in 0..<curveCount {
                var curve = [CGPoint]()
                for _ in 0..<pointsPerCurve {
                    let x = CGFloat(ptr[offset])
                    let y = CGFloat(ptr[offset + 1])
                    curve.append(CGPoint(x: x, y: y))
                    offset += 2
                }
                curves.append(curve)
            }
            return curves
        } catch {
            return nil
        }
    }
    
    private func makeInputArray(request: ARKitLipParsingRequest) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: inputChannels), NSNumber(value: inputHeight), NSNumber(value: inputWidth)], dataType: .float32)
        
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let strides = array.strides.map { $0.intValue }
        
        CVPixelBufferLockBaseAddress(request.pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(request.pixelBuffer, .readOnly) }
        
        guard let sampler = ARKitPixelSampler(pixelBuffer: request.pixelBuffer), strides.count == 4 else {
            throw CocoaError(.coderInvalidValue)
        }
        
        sampler.fillNormalizedRGB(
            into: ptr,
            strides: strides,
            inputWidth: inputWidth,
            inputHeight: inputHeight,
            topLeft: request.topLeft,
            topRight: request.topRight,
            bottomLeft: request.bottomLeft,
            bottomRight: request.bottomRight,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225]
        )
        
        return array
    }
}
