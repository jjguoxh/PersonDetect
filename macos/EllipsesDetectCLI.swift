import Foundation
import CoreML
import Vision
import AVFoundation
import CoreImage
import CoreGraphics

struct Config {
    let modelURL: URL
    let inputURL: URL
    let outputURL: URL
    let confidence: Float
    let debug: Bool
    let anchor: String
    let vOffset: CGFloat
    let coord: String
}

final class EllipsesDetector {
    private let vnModel: VNCoreMLModel
    private let ciContext = CIContext()
    private let confidence: Float
    private let debug: Bool
    private let anchor: String
    private let vOffset: CGFloat
    private let coord: String

    init(modelURL: URL, confidence: Float, debug: Bool, anchor: String, vOffset: CGFloat, coord: String) throws {
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let mlmodel = try MLModel(contentsOf: compiledURL)
        self.vnModel = try VNCoreMLModel(for: mlmodel)
        self.confidence = confidence
        self.debug = debug
        self.anchor = anchor
        self.vOffset = vOffset
        self.coord = coord
    }

    func detectPersons(in pixelBuffer: CVPixelBuffer) throws -> [VNRecognizedObjectObservation] {
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])
        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            return []
        }
        return observations.filter { obs in
            guard obs.confidence >= confidence else { return false }
            if let topLabel = obs.labels.first {
                return topLabel.identifier.lowercased() == "person" || topLabel.identifier.lowercased() == "0" // model may use class id
            }
            return false
        }
    }

    func drawEllipses(on pixelBuffer: CVPixelBuffer, boxes: [VNRecognizedObjectObservation]) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return }
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let ctx = CGContext(data: baseAddress,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: bitmapInfo) else { return }

        // 根据选项决定绘制坐标系：bl(左下角原点) 或 tl(左上角原点)
        let useBL = coord.lowercased() == "bl" || coord.lowercased() == "bottom-left"
        if useBL {
            ctx.translateBy(x: 0, y: CGFloat(height))
            ctx.scaleBy(x: 1.0, y: -1.0)
        }

        ctx.setLineWidth(4)
        ctx.setStrokeColor(CGColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 1.0))
        ctx.setFillColor(CGColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 0.2))

        for obs in boxes {
            // 直接由归一化坐标计算像素矩形，避免 VNImageRectForNormalizedRect 的隐式假设
            let b = obs.boundingBox
            let rect: CGRect
            if useBL {
                // 左下角原点绘制，但许多模型的 boundingBox 原点是“左上角”规范。
                // 因此这里按“原点在左上角”来换算到左下角绘制坐标。
                rect = CGRect(
                    x: CGFloat(b.origin.x) * CGFloat(width),
                    y: (1.0 - CGFloat(b.origin.y) - CGFloat(b.size.height)) * CGFloat(height),
                    width: CGFloat(b.size.width) * CGFloat(width),
                    height: CGFloat(b.size.height) * CGFloat(height)
                )
                
            } else {
                // 左上角原点：y 需做反转并减去高度
                rect = CGRect(
                    x: CGFloat(b.origin.x) * CGFloat(width),
                    y: (1.0 - CGFloat(b.origin.y) - CGFloat(b.size.height)) * CGFloat(height),
                    width: CGFloat(b.size.width) * CGFloat(width),
                    height: CGFloat(b.size.height) * CGFloat(height)
                )
            }
            if debug {
                // 画出检测框，便于定位问题
                ctx.setStrokeColor(CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0))
                ctx.stroke(rect)
                ctx.setStrokeColor(CGColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 1.0))
            }

            let centerX = rect.midX
            // 选择锚点（以最终视觉效果为准）：
            // - BL 绘制模式下，矩形由 TL 数值经上下翻转得到，此时视觉上的 bottom=rect.maxY、top=rect.minY
            // - TL 绘制模式下，视觉上的 bottom=rect.maxY、top=rect.minY（常规）
            let centerY: CGFloat
            switch anchor.lowercased() {
            case "top":
                centerY = useBL ? rect.minY : rect.minY
            case "center":
                centerY = rect.midY
            default: // bottom
                centerY = useBL ? rect.maxY : rect.maxY
            }

            let ellipseWidth = rect.width / 2.0
            let ellipseHeight = max(8.0, rect.height / 8.0)
            let ellipseRect = CGRect(
                x: centerX - ellipseWidth / 2.0,
                y: (centerY + vOffset) - ellipseHeight / 2.0,
                width: ellipseWidth,
                height: ellipseHeight
            )
            ctx.strokeEllipse(in: ellipseRect)
            ctx.fillEllipse(in: ellipseRect)
        }
    }

    func processVideo(inputURL: URL, outputURL: URL) throws {
        let asset = AVAsset(url: inputURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw NSError(domain: "EllipsesDetect", code: -1, userInfo: [NSLocalizedDescriptionKey: "无法找到视频轨道"])
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

        // Prepare writer (ensure output path is clean)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            do { try FileManager.default.removeItem(at: outputURL) } catch {}
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

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        guard writer.startWriting() else {
            throw NSError(domain: "EllipsesDetect", code: -2, userInfo: [NSLocalizedDescriptionKey: "无法开始写入: \(writer.error?.localizedDescription ?? "未知错误")"])
        }
        guard reader.startReading() else {
            throw NSError(domain: "EllipsesDetect", code: -3, userInfo: [NSLocalizedDescriptionKey: "无法开始读取: \(reader.error?.localizedDescription ?? "未知错误")"])
        }

        var started = false
        var frameCount: Int64 = 0
        let frameDuration = CMTime(seconds: 1.0 / fps, preferredTimescale: 600)

        let startTime = CMTime.zero
        writer.startSession(atSourceTime: startTime)

        while reader.status == .reading {
            guard let sampleBuffer = readerOutput.copyNextSampleBuffer() else { break }
            guard let srcPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            if !started { started = true }

            // Create destination pixel buffer from adaptor's pool
            var dstPixelBufferOpt: CVPixelBuffer? = nil
            guard let pool = adaptor.pixelBufferPool else {
                throw NSError(domain: "EllipsesDetect", code: -4, userInfo: [NSLocalizedDescriptionKey: "无法创建像素缓冲池"])
            }
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dstPixelBufferOpt)
            guard let dstPixelBuffer = dstPixelBufferOpt else { continue }

            // Render source to destination
            let srcImage = CIImage(cvPixelBuffer: srcPixelBuffer)
            ciContext.render(srcImage, to: dstPixelBuffer)

            // Detect and draw
            let boxes = try detectPersons(in: dstPixelBuffer)
            drawEllipses(on: dstPixelBuffer, boxes: boxes)

            let pts = CMTimeMultiply(frameDuration, multiplier: Int32(frameCount))

            while !writerInput.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
            if !adaptor.append(dstPixelBuffer, withPresentationTime: pts) {
                throw NSError(domain: "EllipsesDetect", code: -5, userInfo: [NSLocalizedDescriptionKey: "写入帧失败: \(writer.error?.localizedDescription ?? "未知错误")"])
            }
            frameCount += 1
            if frameCount % 30 == 0 {
                print("已处理帧数: \(frameCount)")
            }
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            if writer.status == .completed {
                print("视频处理完成，输出: \(outputURL.path)")
            } else {
                print("写入结束异常: \(writer.error?.localizedDescription ?? "未知错误")")
            }
        }
    }
}

func parseArgs() -> Config {
    let args = CommandLine.arguments
    func value(for flag: String, default def: String? = nil) -> String? {
        // 选择最后一次出现的参数作为有效值，便于覆盖默认值
        if let idx = args.lastIndex(of: flag), idx + 1 < args.count { return args[idx + 1] }
        return def
    }
    let modelPath = value(for: "--model", default: "macos/model/yolo11x.mlpackage")!
    let inputPath = value(for: "--input", default: "video/market-square.mp4")!
    let outputPath = value(for: "--output", default: "result/result_with_ellipses.mp4")!
    let confStr = value(for: "--confidence", default: "0.3")!
    let conf = Float(confStr) ?? 0.3
    let debug = args.contains("--debug")
    let anchor = value(for: "--anchor", default: "bottom")!.lowercased()
    let vOffsetStr = value(for: "--voffset", default: "0")!
    let vOffset = CGFloat(Double(vOffsetStr) ?? 0.0)
    let coord = value(for: "--coord", default: "bl")!.lowercased()
    return Config(modelURL: URL(fileURLWithPath: modelPath),
                  inputURL: URL(fileURLWithPath: inputPath),
                  outputURL: URL(fileURLWithPath: outputPath),
                  confidence: conf,
                  debug: debug,
                  anchor: anchor,
                  vOffset: vOffset,
                  coord: coord)
}

do {
    let cfg = parseArgs()
    print("模型路径: \(cfg.modelURL.path)")
    print("输入视频: \(cfg.inputURL.path)")
    print("输出视频: \(cfg.outputURL.path)")
    let detector = try EllipsesDetector(modelURL: cfg.modelURL, confidence: cfg.confidence, debug: cfg.debug, anchor: cfg.anchor, vOffset: cfg.vOffset, coord: cfg.coord)
    try detector.processVideo(inputURL: cfg.inputURL, outputURL: cfg.outputURL)
    // Wait briefly to ensure writer completes asynchronous finishWriting closure
    RunLoop.current.run(until: Date().addingTimeInterval(1.0))
} catch {
    fputs("错误: \(error)\n", stderr)
    exit(1)
}