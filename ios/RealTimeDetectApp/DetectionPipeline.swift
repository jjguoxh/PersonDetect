import Foundation
import Vision
import CoreML
import CoreGraphics

// MARK: - Object Detection via VNRecognizedObjectObservation
final class ObjectDetectPipeline {
    private let vnModel: VNCoreMLModel
    private let sequenceHandler = VNSequenceRequestHandler()
    private let objectRequest: VNCoreMLRequest
    init(modelName: String) throws {
        // 优先在 iOS 目录的 Models 子目录查找
        let config = MLModelConfiguration()
        config.computeUnits = .all // GPU/ANE 优先
        if let url = Self.resolveModelURL(modelName: modelName, preferredExt: "mlmodelc") {
            let mlmodel = try MLModel(contentsOf: url, configuration: config)
            self.vnModel = try VNCoreMLModel(for: mlmodel)
        } else if let packageURL = Self.resolveModelURL(modelName: modelName, preferredExt: "mlpackage") {
            let compiledURL = try MLModel.compileModel(at: packageURL)
            let mlmodel = try MLModel(contentsOf: compiledURL, configuration: config)
            self.vnModel = try VNCoreMLModel(for: mlmodel)
        } else {
            throw NSError(domain: "ObjectDetectPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "未找到模型资源: \(modelName) (查找: Models/|Bundle)"])
        }
        let req = VNCoreMLRequest(model: vnModel)
        req.imageCropAndScaleOption = .scaleFill
        req.usesCPUOnly = false
        req.preferBackgroundProcessing = true
        self.objectRequest = req
    }
    func detectObjects(in pixelBuffer: CVPixelBuffer, minConfidence: Float) throws -> [VNRecognizedObjectObservation] {
        try sequenceHandler.perform([objectRequest], on: pixelBuffer)
        guard let observations = objectRequest.results as? [VNRecognizedObjectObservation] else { return [] }
        return observations.filter { $0.confidence >= minConfidence }
    }
    private static func resolveModelURL(modelName: String, preferredExt: String) -> URL? {
        let bundle = Bundle.main
        let subdirs: [String?] = ["Models", "RealTimeDetectApp", nil]
        for sub in subdirs {
            if let sub = sub, let url = bundle.url(forResource: modelName, withExtension: preferredExt, subdirectory: sub) {
                return url
            }
        }
        return bundle.url(forResource: modelName, withExtension: preferredExt)
    }
}

// MARK: - Pose Detection (YOLO-Pose decode)
struct Keypoint { var x: CGFloat; var y: CGFloat; var conf: Float }
struct Pose { var bbox: CGRect; var score: Float; var keypoints: [Keypoint] }

final class PoseDetectPipeline {
    private let vnModel: VNCoreMLModel
    private let sequenceHandler = VNSequenceRequestHandler()
    private let poseRequest: VNCoreMLRequest
    init(modelName: String) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // GPU/ANE 优先
        if let url = Self.resolveModelURL(modelName: modelName, preferredExt: "mlmodelc") {
            let mlmodel = try MLModel(contentsOf: url, configuration: config)
            self.vnModel = try VNCoreMLModel(for: mlmodel)
        } else if let packageURL = Self.resolveModelURL(modelName: modelName, preferredExt: "mlpackage") {
            let compiledURL = try MLModel.compileModel(at: packageURL)
            let mlmodel = try MLModel(contentsOf: compiledURL, configuration: config)
            self.vnModel = try VNCoreMLModel(for: mlmodel)
        } else {
            throw NSError(domain: "PoseDetectPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "未找到模型资源: \(modelName) (查找: Models/|Bundle)"])
        }
        let req = VNCoreMLRequest(model: vnModel)
        req.imageCropAndScaleOption = .scaleFit
        req.usesCPUOnly = false
        req.preferBackgroundProcessing = true
        self.poseRequest = req
    }

    private static func resolveModelURL(modelName: String, preferredExt: String) -> URL? {
        let bundle = Bundle.main
        let subdirs: [String?] = ["Models", "RealTimeDetectApp", nil]
        for sub in subdirs {
            if let sub = sub, let url = bundle.url(forResource: modelName, withExtension: preferredExt, subdirectory: sub) {
                return url
            }
        }
        return bundle.url(forResource: modelName, withExtension: preferredExt)
    }

    func predictPoses(in pixelBuffer: CVPixelBuffer, confidence: Float, kpThreshold: Float, maxDet: Int = 5) throws -> [Pose] {
        try sequenceHandler.perform([poseRequest], on: pixelBuffer)
        guard let observations = poseRequest.results as? [VNCoreMLFeatureValueObservation],
              let feature = observations.first?.featureValue.multiArrayValue else {
            return []
        }
        return decodePoses(from: feature, inputW: 640, inputH: 640, confidence: confidence, kpThreshold: kpThreshold, maxDet: maxDet)
    }

    private func decodePoses(from feature: MLMultiArray, inputW: Int, inputH: Int, confidence: Float, kpThreshold: Float, maxDet: Int) -> [Pose] {
        let shape = feature.shape.map { Int(truncating: $0) }
        guard shape.count == 3 else { return [] }
        let axisC = shape[1] <= 70 ? 1 : 2
        let axisA = axisC == 1 ? 2 : 1
        let C = shape[axisC]
        let A = shape[axisA]
        let strides = feature.strides.map { Int(truncating: $0) }
        let strideC = strides[axisC]
        let strideA = strides[axisA]
        func valueAt(_ c: Int, _ a: Int) -> Float {
            let idx = 0 * strides[0] + c * strideC + a * strideA
            switch feature.dataType {
            case .float32:
                return feature.dataPointer.assumingMemoryBound(to: Float.self)[idx]
            case .double:
                return Float(feature.dataPointer.assumingMemoryBound(to: Double.self)[idx])
            case .float16:
                let u16 = feature.dataPointer.assumingMemoryBound(to: UInt16.self)[idx]
                return halfToFloat(u16)
            default:
                return 0
            }
        }
        let expectedKPDims = 5 + 17 * 3
        let numClasses = C > expectedKPDims ? (C - expectedKPDims) : 0
        let kpStart = 5 + numClasses
        let hasKeypoints = C >= expectedKPDims
        var candidates: [Pose] = []
        for a in 0..<A {
            let objRaw = valueAt(4, a)
            let objConf = (objRaw < 0 || objRaw > 1) ? sigmoid(objRaw) : objRaw
            if objConf < confidence { continue }
            let cxRaw = valueAt(0, a)
            let cyRaw = valueAt(1, a)
            let wRaw  = valueAt(2, a)
            let hRaw  = valueAt(3, a)
            let maxVal = max(max(cxRaw, cyRaw), max(wRaw, hRaw))
            let isBoxNormalized = maxVal <= 1.5
            let cx = CGFloat(isBoxNormalized ? cxRaw * Float(inputW) : cxRaw)
            let cy = CGFloat(isBoxNormalized ? cyRaw * Float(inputH) : cyRaw)
            let w  = CGFloat(isBoxNormalized ? wRaw  * Float(inputW) : wRaw)
            let h  = CGFloat(isBoxNormalized ? hRaw  * Float(inputH) : hRaw)
            if w < 32 || h < 32 { continue }
            let x = max(0, cx - w / 2.0)
            let y = max(0, cy - h / 2.0)
            var kps: [Keypoint] = []
            if hasKeypoints {
                for j in 0..<17 {
                    let base = kpStart + j * 3
                    if base + 2 >= C { break }
                    let kxRaw = valueAt(base + 0, a)
                    let kyRaw = valueAt(base + 1, a)
                    let kcRaw = valueAt(base + 2, a)
                    let kc    = (kcRaw < 0 || kcRaw > 1) ? sigmoid(kcRaw) : kcRaw
                    let isRelToBox = abs(kxRaw) <= 1.5 && abs(kyRaw) <= 1.5
                    var kx: CGFloat
                    var ky: CGFloat
                    if isRelToBox {
                        kx = cx + CGFloat(kxRaw) * w
                        ky = cy + CGFloat(kyRaw) * h
                    } else {
                        let isKPNormalized = max(abs(kxRaw), abs(kyRaw)) <= 1.5
                        kx = CGFloat(isKPNormalized ? kxRaw * Float(inputW) : kxRaw)
                        ky = CGFloat(isKPNormalized ? kyRaw * Float(inputH) : kyRaw)
                    }
                    let kxClamped = min(max(kx, 0.0), CGFloat(inputW - 1))
                    let kyClamped = min(max(ky, 0.0), CGFloat(inputH - 1))
                    kps.append(Keypoint(x: kxClamped, y: kyClamped, conf: kc))
                }
            }
            if hasKeypoints {
                let validCount = kps.reduce(0) { $1.conf >= kpThreshold ? $0 + 1 : $0 }
                if validCount < 6 { continue }
            }
            candidates.append(Pose(bbox: CGRect(x: x, y: y, width: w, height: h), score: objConf, keypoints: kps))
        }
        candidates.sort { $0.score > $1.score }
        var result: [Pose] = []
        let iouThresh: Float = 0.5
        for p in candidates {
            var keep = true
            for r in result {
                if iou(p.bbox, r.bbox) > iouThresh { keep = false; break }
            }
            if keep {
                result.append(p)
                if result.count >= maxDet { break }
            }
        }
        return result
    }

    private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }
    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = Float(inter.width * inter.height)
        if interArea <= 0 { return 0 }
        let areaA = Float(a.width * a.height)
        let areaB = Float(b.width * b.height)
        let denom = areaA + areaB - interArea
        if denom <= 0 { return 0 }
        return interArea / denom
    }
    private func halfToFloat(_ h: UInt16) -> Float {
        #if swift(>=5.5)
        return Float(Float16(bitPattern: h))
        #else
        // 简易IEEE754 half->float转换（旧版Swift备用）
        let s = (h >> 15) & 0x1
        let e = Int((h >> 10) & 0x1F)
        let f = Int(h & 0x3FF)
        if e == 0 {
            return Float(pow(2.0, -14.0)) * Float(f) / 1024.0 * (s == 0 ? 1.0 : -1.0)
        } else if e == 31 {
            return f == 0 ? (s == 0 ? Float.infinity : -Float.infinity) : Float.nan
        }
        let value = Float(pow(2.0, Float(e - 15))) * (1.0 + Float(f) / 1024.0)
        return s == 0 ? value : -value
        #endif
    }
}