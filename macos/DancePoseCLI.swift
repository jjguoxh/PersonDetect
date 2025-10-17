import Foundation
import CoreML
import Vision
import AVFoundation
import CoreImage
import CoreGraphics
import CoreText
import AppKit

struct PoseConfig {
    let modelURL: URL
    let inputURL: URL?
    let outputURL: URL?
    let dir: URL?
    let outdir: URL?
    let confidence: Float
    let keypointThreshold: Float
    let debug: Bool
}

struct Keypoint {
    var x: CGFloat
    var y: CGFloat
    var conf: Float
}

struct Pose {
    var bbox: CGRect
    var score: Float
    var keypoints: [Keypoint] // 17
}

final class DancePoseDetector {
    private let vnModel: VNCoreMLModel
    private let ciContext = CIContext()
    private let confidence: Float
    private let keypointThreshold: Float
    private let debug: Bool
    private var printedFeatureInfo = false
    private var featureInfoText: String? = nil

    init(modelURL: URL, confidence: Float, keypointThreshold: Float, debug: Bool) throws {
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let mlmodel = try MLModel(contentsOf: compiledURL)
        self.vnModel = try VNCoreMLModel(for: mlmodel)
        self.confidence = confidence
        self.keypointThreshold = keypointThreshold
        self.debug = debug
    }

    // YOLO Pose output: MLMultiArray of shape [1, 56, 8400]
    // 56 dims per anchor: [cx, cy, w, h, conf, kp0_x, kp0_y, kp0_conf, ..., kp16_x, kp16_y, kp16_conf]
    private func decodePoses(from feature: MLMultiArray, inputW: Int, inputH: Int, maxDet: Int = 10) -> [Pose] {
        let shape = feature.shape.map { Int(truncating: $0) }
        guard shape.count == 3 else { return [] }
        // Determine axes: some exports use [1, C, A], others [1, A, C]
        let axisC = shape[1] <= 70 ? 1 : 2
        let axisA = axisC == 1 ? 2 : 1
        let C = shape[axisC]
        let A = shape[axisA]
        let strides = feature.strides.map { Int(truncating: $0) }
        let strideC = strides[axisC]
        let strideA = strides[axisA]
        // Prepare overlay text once
        if featureInfoText == nil {
            featureInfoText = "feature.shape=\(shape), C=\(C), A=\(A), dataType=\(feature.dataType)"
        }
        if debug && !printedFeatureInfo {
            printedFeatureInfo = true
            print("[DEBUG] feature.shape=\(shape), axes(C=\(axisC),A=\(axisA)), C=\(C), A=\(A), strides=\(strides), dataType=\(feature.dataType)")
            var sample: [Float] = []
            for c in 0..<min(C, 32) { sample.append(valueAtRaw(feature: feature, strideC: strideC, strideA: strideA, c, 0)) }
            print("[DEBUG] first anchor first \(min(C,32)) dims: \(sample.map{String(format: "%.3f", $0)}.joined(separator: ", "))")
        }
        // Generic indexer using detected axes
        func valueAt(_ c: Int, _ a: Int) -> Float {
            let idx = 0 * strides[0] + c * strideC + a * strideA
            switch feature.dataType {
            case .float32:
                return feature.dataPointer.assumingMemoryBound(to: Float.self)[idx]
            case .double:
                return Float(feature.dataPointer.assumingMemoryBound(to: Double.self)[idx])
            case .float16:
                let u16 = feature.dataPointer.assumingMemoryBound(to: UInt16.self)[idx]
                return Float(halfToFloat(u16))
            default:
                return 0
            }
        }
        func valueAtRaw(feature: MLMultiArray, strideC: Int, strideA: Int, _ c: Int, _ a: Int) -> Float {
            let idx = 0 * strides[0] + c * strideC + a * strideA
            switch feature.dataType {
            case .float32:
                return feature.dataPointer.assumingMemoryBound(to: Float.self)[idx]
            case .double:
                return Float(feature.dataPointer.assumingMemoryBound(to: Double.self)[idx])
            case .float16:
                let u16 = feature.dataPointer.assumingMemoryBound(to: UInt16.self)[idx]
                return Float(halfToFloat(u16))
            default:
                return 0
            }
        }

        // Classes count if present
        let expectedKPDims = 5 + 17 * 3
        let numClasses = C > expectedKPDims ? (C - expectedKPDims) : 0
        let kpStart = 5 + numClasses
        let hasKeypoints = C >= expectedKPDims
        if debug && !hasKeypoints {
            print("[WARN] Model output C=\(C) < \(expectedKPDims), 未检测到关键点通道，可能是检测模型而非姿态模型")
        }

        var candidates: [Pose] = []
        for a in 0..<A {
            let objRaw = valueAt(4, a)
            let objConf = (objRaw < 0 || objRaw > 1) ? sigmoid(objRaw) : objRaw
            if objConf < confidence && !debug { continue }

            // Heuristics: some exports output normalized [0,1], others pixel space [0..input]
            let cxRaw = valueAt(0, a)
            let cyRaw = valueAt(1, a)
            let wRaw  = valueAt(2, a)
            let hRaw  = valueAt(3, a)
            let isBoxNormalized = max(cxRaw, cyRaw, wRaw, hRaw) <= 1.5
            let cx = CGFloat(isBoxNormalized ? cxRaw * Float(inputW) : cxRaw)
            let cy = CGFloat(isBoxNormalized ? cyRaw * Float(inputH) : cyRaw)
            let w  = CGFloat(isBoxNormalized ? wRaw  * Float(inputW) : wRaw)
            let h  = CGFloat(isBoxNormalized ? hRaw  * Float(inputH) : hRaw)

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
                    // Some exports encode keypoints relative to bbox center + size
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
                    let kxClamped = min(max(kx, 0), CGFloat(inputW - 1))
                    let kyClamped = min(max(ky, 0), CGFloat(inputH - 1))
                    kps.append(Keypoint(x: kxClamped, y: kyClamped, conf: kc))
                }
            }
            candidates.append(Pose(bbox: CGRect(x: x, y: y, width: w, height: h), score: objConf, keypoints: kps))
        }
        // Sort by score desc
        candidates.sort { $0.score > $1.score }
        // NMS
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

    // Inference with Vision using scaleFit letterbox mapping to 640x640
    private func predictPoses(in pixelBuffer: CVPixelBuffer) throws -> [Pose] {
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
              let feature = observations.first?.featureValue.multiArrayValue else {
            return []
        }
        return decodePoses(from: feature, inputW: 640, inputH: 640, maxDet: 10)
    }

    // Helpers for decode
    private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
    // IoU for NMS
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
        // Fallback: basic IEEE 754 half->float conversion
        let s = (h >> 15) & 0x1
        var e = Int((h >> 10) & 0x1F)
        var f = Int(h & 0x3FF)
        var value: Float
        if e == 0 {
            if f == 0 {
                value = 0
            } else {
                value = Float(pow(2.0, -14)) * Float(f) / 1024.0
            }
        } else if e == 31 {
            value = f == 0 ? Float.infinity : Float.nan
        } else {
            value = Float(pow(2.0, Float(e - 15))) * (1.0 + Float(f) / 1024.0)
        }
        return s == 0 ? value : -value
        #endif
    }

    func drawPoses(on pixelBuffer: CVPixelBuffer, poses: [Pose]) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return }
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let ctx = CGContext(data: baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else { return }

        // Transform to BL-origin for consistent visual interpretation
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)

        // Mapping from model(640x640 with scaleFit letterbox) to original frame
        let w = CGFloat(width)
        let h = CGFloat(height)
        let sFitPre = min(640.0 / w, 640.0 / h)
        let padX = (640.0 - w * sFitPre) / 2.0
        let padY = (640.0 - h * sFitPre) / 2.0
        func mapToOriginal(_ mx: CGFloat, _ my: CGFloat) -> CGPoint {
            let ox = (mx - padX) / sFitPre
            let oy = (my - padY) / sFitPre
            return CGPoint(x: ox, y: oy)
        }
        func clampToFrame(_ p: CGPoint) -> CGPoint {
            return CGPoint(x: min(max(p.x, 0), w-1), y: min(max(p.y, 0), h-1))
        }

        // Skeleton connections
        let connections: [(Int, Int)] = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,11),(6,12),(11,12),
            (5,7),(7,9),(6,8),(8,10),
            (11,13),(13,15),(12,14),(14,16)
        ]
        let names = [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
            "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
        ]

        for pose in poses {
            // Draw bbox if debug
            if debug {
                ctx.setStrokeColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
                let tl = clampToFrame(mapToOriginal(pose.bbox.minX, pose.bbox.minY))
                let br = clampToFrame(mapToOriginal(pose.bbox.maxX, pose.bbox.maxY))
                let r = CGRect(x: tl.x, y: tl.y, width: br.x - tl.x, height: br.y - tl.y)
                ctx.stroke(r)
            }

            // Draw skeleton
            ctx.setLineWidth(4)
            ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))

            // Keypoint circles and labels
            for (i, kp) in pose.keypoints.enumerated() {
                if !debug && kp.conf < keypointThreshold { continue }
                let p = clampToFrame(mapToOriginal(kp.x, kp.y))
                let px = p.x
                let py = p.y
                let color: CGColor
                if i <= 4 { // head
                    color = CGColor(red: 1, green: 0.5, blue: 0, alpha: 1)
                } else if i <= 10 { // upper limbs
                    color = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                } else { // lower limbs
                    color = CGColor(red: 1, green: 1, blue: 0, alpha: 1)
                }
                ctx.setFillColor(color)
                ctx.fillEllipse(in: CGRect(x: px-6, y: py-6, width: 12, height: 12))
                // white border
                ctx.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
                ctx.strokeEllipse(in: CGRect(x: px-8, y: py-8, width: 16, height: 16))
                // label
                let label = names[i]
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 12),
                    .foregroundColor: NSColor.white
                ]
                let text = NSAttributedString(string: label, attributes: attrs)
                let line = CTLineCreateWithAttributedString(text)
                ctx.textPosition = CGPoint(x: px + 10, y: py + 10)
                CTLineDraw(line, ctx)
            }

            // Draw connections (ignore threshold in debug)
            ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
            for (a, b) in connections {
                let kpa = pose.keypoints[a]
                let kpb = pose.keypoints[b]
                if !debug && (kpa.conf < keypointThreshold || kpb.conf < keypointThreshold) { continue }
                let pa = clampToFrame(mapToOriginal(kpa.x, kpa.y))
                let pb = clampToFrame(mapToOriginal(kpb.x, kpb.y))
                ctx.move(to: pa)
                ctx.addLine(to: pb)
                ctx.strokePath()
            }

            // Forced raw keypoint overlay in debug mode to visualize coordinate hypotheses
            if debug {
                for kp in pose.keypoints {
                    // Hypothesis A: model-space mapped via letterbox (current assumption) - RED
                    let pA = clampToFrame(mapToOriginal(kp.x, kp.y))
                    ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 0.8))
                    ctx.fill(CGRect(x: pA.x - 3, y: pA.y - 3, width: 6, height: 6))

                    // Hypothesis B: normalized to original frame [0,1] - BLUE
                    let pB = clampToFrame(CGPoint(x: kp.x * w, y: kp.y * h))
                    ctx.setFillColor(CGColor(red: 0, green: 0.5, blue: 1, alpha: 0.8))
                    ctx.fill(CGRect(x: pB.x - 3, y: pB.y - 3, width: 6, height: 6))

                    // Hypothesis C: raw direct coordinates (no mapping) - MAGENTA
                    let pC = clampToFrame(CGPoint(x: kp.x, y: kp.y))
                    ctx.setFillColor(CGColor(red: 1, green: 0, blue: 1, alpha: 0.8))
                    ctx.fill(CGRect(x: pC.x - 3, y: pC.y - 3, width: 6, height: 6))
                }
                // Legend block in top-left
                ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.3))
                ctx.fill(CGRect(x: 10, y: h - 90, width: 340, height: 80))
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 13),
                    .foregroundColor: NSColor.white
                ]
                let legend1 = NSAttributedString(string: "Raw overlay A: mapToOriginal (red)", attributes: attrs)
                let legend2 = NSAttributedString(string: "Raw overlay B: normalized to frame (blue)", attributes: attrs)
                let legend3 = NSAttributedString(string: "Raw overlay C: raw direct (magenta)", attributes: attrs)
                let l1 = CTLineCreateWithAttributedString(legend1)
                let l2 = CTLineCreateWithAttributedString(legend2)
                let l3 = CTLineCreateWithAttributedString(legend3)
                ctx.textPosition = CGPoint(x: 20, y: h - 20)
                CTLineDraw(l1, ctx)
                ctx.textPosition = CGPoint(x: 20, y: h - 40)
                CTLineDraw(l2, ctx)
                ctx.textPosition = CGPoint(x: 20, y: h - 60)
                CTLineDraw(l3, ctx)
            }
        }

        // Draw feature info text if available (top-left)
        if let info = featureInfoText {
            ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.5))
            ctx.fill(CGRect(x: 10, y: h - 30, width: min(w - 20, CGFloat(780)), height: 20))
            let attrs2: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 14),
                .foregroundColor: NSColor.white
            ]
            let s = NSAttributedString(string: info, attributes: attrs2)
            let line = CTLineCreateWithAttributedString(s)
            ctx.textPosition = CGPoint(x: 20, y: h - 20)
            CTLineDraw(line, ctx)
        }
    }

    func processVideo(inputURL: URL, outputURL: URL) throws {
        let asset = AVAsset(url: inputURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw NSError(domain: "DancePose", code: -1, userInfo: [NSLocalizedDescriptionKey: "无法找到视频轨道"])
        }

        let reader = try AVAssetReader(asset: asset)
        let readerOutputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: readerOutputSettings)
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)

        let naturalSize = track.naturalSize.applying(track.preferredTransform)
        let width = Int(abs(naturalSize.width))
        let height = Int(abs(naturalSize.height))
        let fps = max(1.0, Double(track.nominalFrameRate))

        // Prepare writer
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try? FileManager.default.removeItem(at: outputURL)
        }
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        writerInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: writerInput,
                                                           sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ])
        writer.add(writerInput)

        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        guard writer.startWriting() else {
            throw NSError(domain: "DancePose", code: -2, userInfo: [NSLocalizedDescriptionKey: "无法开始写入: \(writer.error?.localizedDescription ?? "未知错误")"])
        }
        guard reader.startReading() else {
            throw NSError(domain: "DancePose", code: -3, userInfo: [NSLocalizedDescriptionKey: "无法开始读取: \(reader.error?.localizedDescription ?? "未知错误")"])
        }

        var frameCount: Int64 = 0
        var totalPersons: Int64 = 0
        let frameDuration = CMTime(seconds: 1.0 / fps, preferredTimescale: 600)
        let processStart = CFAbsoluteTimeGetCurrent()
        writer.startSession(atSourceTime: .zero)

        while reader.status == .reading {
            guard let sampleBuffer = readerOutput.copyNextSampleBuffer() else { break }
            guard let srcPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            var dstPixelBufferOpt: CVPixelBuffer? = nil
            guard let pool = adaptor.pixelBufferPool else {
                throw NSError(domain: "DancePose", code: -4, userInfo: [NSLocalizedDescriptionKey: "无法创建像素缓冲池"])
            }
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dstPixelBufferOpt)
            guard let dstPixelBuffer = dstPixelBufferOpt else { continue }

            let srcImage = CIImage(cvPixelBuffer: srcPixelBuffer)
            ciContext.render(srcImage, to: dstPixelBuffer)

            // Predict and draw poses
            let poses = try predictPoses(in: dstPixelBuffer)
            if debug {
                print("[POSE] frame=\(frameCount) persons=\(poses.count)")
                let names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
                for (pi, p) in poses.enumerated() {
                    let bboxDesc = String(format: "(x:%.0f,y:%.0f,w:%.0f,h:%.0f)", Double(p.bbox.origin.x), Double(p.bbox.origin.y), Double(p.bbox.size.width), Double(p.bbox.size.height))
                    print("[POSE]   person \(pi+1) bbox=\(bboxDesc), score=\(String(format: "%.2f", Double(p.score)))")
                    let kpStrs = p.keypoints.enumerated().map { (idx, kp) in
                        let name = idx < names.count ? names[idx] : "kp\(idx)"
                        return "\(name):(\(String(format: "%.1f", Double(kp.x))),\(String(format: "%.1f", Double(kp.y))))[\(String(format: "%.2f", Double(kp.conf)))]"
                    }
                    print("[POSE]   keypoints: \(kpStrs.joined(separator: ", "))")
                    if pi >= 1 { break } // 限制日志输出前两个人
                }
            }
            if frameCount == 0 {
                print("[INFO] feature overlay: \(featureInfoText ?? "(unknown)")")
            }
            drawPoses(on: dstPixelBuffer, poses: poses)
            totalPersons += Int64(poses.count)

            // Overlay simple text: frame and person count
            CVPixelBufferLockBaseAddress(dstPixelBuffer, [])
            if let ctx2 = CGContext(data: CVPixelBufferGetBaseAddress(dstPixelBuffer), width: CVPixelBufferGetWidth(dstPixelBuffer), height: CVPixelBufferGetHeight(dstPixelBuffer), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(dstPixelBuffer), space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue) {
                ctx2.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.4))
                ctx2.fill(CGRect(x: 10, y: 10, width: 900, height: 110))
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 24),
                    .foregroundColor: NSColor.green
                ]
                let s1 = NSAttributedString(string: "Frame: \(frameCount)", attributes: attrs)
                let s2 = NSAttributedString(string: "Persons: \(poses.count)", attributes: attrs)
                let l1 = CTLineCreateWithAttributedString(s1)
                let l2 = CTLineCreateWithAttributedString(s2)
                ctx2.textPosition = CGPoint(x: 20, y: CGFloat(CVPixelBufferGetHeight(dstPixelBuffer)) - 26)
                CTLineDraw(l1, ctx2)
                ctx2.textPosition = CGPoint(x: 20, y: CGFloat(CVPixelBufferGetHeight(dstPixelBuffer)) - 52)
                CTLineDraw(l2, ctx2)
                // Additional feature info text (C/A/shape/dataType) below, left-aligned for visibility
                let infoText = featureInfoText ?? "feature.shape=?, C=?, A=?, dataType=?"
                let attrsInfo: [NSAttributedString.Key: Any] = [
                    .font: NSFont.boldSystemFont(ofSize: 18),
                    .foregroundColor: NSColor.yellow
                ]
                let sInfo = NSAttributedString(string: infoText, attributes: attrsInfo)
                let lInfo = CTLineCreateWithAttributedString(sInfo)
                ctx2.textPosition = CGPoint(x: 20, y: CGFloat(CVPixelBufferGetHeight(dstPixelBuffer)) - 78)
                CTLineDraw(lInfo, ctx2)
            }
            CVPixelBufferUnlockBaseAddress(dstPixelBuffer, [])

            let pts = CMTimeMultiply(frameDuration, multiplier: Int32(frameCount))
            while !writerInput.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
            _ = adaptor.append(dstPixelBuffer, withPresentationTime: pts)
            frameCount += 1
            if frameCount % 30 == 0 {
                print("已处理帧数: \(frameCount), 当前人数: \(poses.count)")
            }
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let processTime = CFAbsoluteTimeGetCurrent() - processStart
            let avgFPS = Double(frameCount) / max(processTime, 0.0001)
            let avgPersonsPerFrame = Double(totalPersons) / Double(max(frameCount, 1))
            if writer.status == .completed {
                let formattedProcess = String(format: "%.2f", processTime)
                let formattedFPS = String(format: "%.2f", avgFPS)
                let formattedPersons = String(format: "%.2f", avgPersonsPerFrame)
                print("视频统计 - 总帧数: \(frameCount), 总耗时: \(formattedProcess)s, 平均FPS: \(formattedFPS), 每帧平均人数: \(formattedPersons)")
                print("视频处理完成，输出: \(outputURL.path)")
            } else {
                print("写入结束异常: \(writer.error?.localizedDescription ?? "未知错误")")
            }
        }
    }
}

func parsePoseArgs() -> PoseConfig {
    let args = CommandLine.arguments
    func value(for flag: String, default def: String? = nil) -> String? {
        if let idx = args.lastIndex(of: flag), idx + 1 < args.count { return args[idx + 1] }
        return def
    }
    let modelPath = value(for: "--model", default: "macos/model/yolon-pose.mlpackage")!
    let inputPath = value(for: "--input")
    let outputPath = value(for: "--output")
    let dirPath = value(for: "--dir", default: "dance")
    let outdirPath = value(for: "--outdir", default: "runs/final_dance_pose")
    let confStr = value(for: "--confidence", default: "0.3")!
    let conf = Float(confStr) ?? 0.3
    let kpStr = value(for: "--kp", default: "0.3")!
    let kp = Float(kpStr) ?? 0.3
    let debug = args.contains("--debug")
    return PoseConfig(modelURL: URL(fileURLWithPath: modelPath),
                      inputURL: inputPath != nil ? URL(fileURLWithPath: inputPath!) : nil,
                      outputURL: outputPath != nil ? URL(fileURLWithPath: outputPath!) : nil,
                      dir: dirPath != nil ? URL(fileURLWithPath: dirPath!) : nil,
                      outdir: outdirPath != nil ? URL(fileURLWithPath: outdirPath!) : nil,
                      confidence: conf,
                      keypointThreshold: kp,
                      debug: debug)
}

func processAllVideos(detector: DancePoseDetector, dir: URL, outdir: URL, confidence: Float) throws {
    let fm = FileManager.default
    guard let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil, options: []) else {
        throw NSError(domain: "DancePose", code: -10, userInfo: [NSLocalizedDescriptionKey: "找不到目录: \(dir.path)"])
    }
    try fm.createDirectory(at: outdir, withIntermediateDirectories: true)
    let videos = items.filter { ["mp4", "avi", "mov", "mkv"].contains($0.pathExtension.lowercased()) }
    print("找到 \(videos.count) 个视频文件:")
    for (idx, v) in videos.enumerated() { print("\(idx+1). \(v.lastPathComponent)") }
    for (idx, v) in videos.enumerated() {
        print("\n处理第 \(idx+1)/\(videos.count) 个视频...")
        let name = v.deletingPathExtension().lastPathComponent
        let out = outdir.appendingPathComponent("\(name)_pose_result.mp4")
        do {
            try detector.processVideo(inputURL: v, outputURL: out)
        } catch {
            print("处理视频 \(v.path) 时出错: \(error)")
        }
    }
    print("\n批量处理完成!")
}

func cameraModeStub() {
    print("摄像头模式：与 Python final_dance_pose_detector.py 保持一致，这里仅提供占位提示。")
    print("当前未实现实时摄像头处理；可后续按需补充 AVCaptureSession 管线。")
}

do {
    let cfg = parsePoseArgs()
    print("姿态模型路径: \(cfg.modelURL.path)")
    let detector = try DancePoseDetector(modelURL: cfg.modelURL, confidence: cfg.confidence, keypointThreshold: cfg.keypointThreshold, debug: cfg.debug)
    if CommandLine.arguments.contains("--camera") {
        cameraModeStub()
    } else if let input = cfg.inputURL, let output = cfg.outputURL {
        print("输入视频: \(input.path)")
        print("输出视频: \(output.path)")
        try detector.processVideo(inputURL: input, outputURL: output)
        RunLoop.current.run(until: Date().addingTimeInterval(1.0))
    } else if let dir = cfg.dir, let outdir = cfg.outdir {
        try processAllVideos(detector: detector, dir: dir, outdir: outdir, confidence: cfg.confidence)
        RunLoop.current.run(until: Date().addingTimeInterval(1.0))
    } else {
        print("请提供 --input/--output 处理单个视频，或提供 --dir/--outdir 进行批量处理，或使用 --camera 进入占位摄像头模式")
        exit(2)
    }
} catch {
    fputs("错误: \(error)\n", stderr)
    exit(1)
}