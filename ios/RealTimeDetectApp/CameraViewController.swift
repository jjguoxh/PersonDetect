import Foundation
import AVFoundation
import Vision
import CoreML
import UIKit

final class CameraViewController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let overlayLayer = CALayer()
    var mode: DetectMode = .objects
    var confidence: Float = 0.5
    var kpThreshold: Float = 0.3

    @Published var statusText: String = ""

    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.sample.queue")

    // Pipelines
    private var objectPipeline: ObjectDetectPipeline?
    private var posePipeline: PoseDetectPipeline?

    // Fallback: Apple Human Body Pose
    private var applePoseRequest: VNDetectHumanBodyPoseRequest?
    private let visionSequence = VNSequenceRequestHandler()

    // Aggregated overlay layers (GPU friendly)
    private let ellipsesLayer = CAShapeLayer()
    private let poseLinesLayer = CAShapeLayer()
    private let poseCirclesLayer = CAShapeLayer()

    override init() {
        super.init()
        overlayLayer.frame = .zero
        overlayLayer.masksToBounds = true
        overlayLayer.contentsScale = UIScreen.main.scale

        // Configure aggregated layers once
        ellipsesLayer.fillColor = UIColor.clear.cgColor
        ellipsesLayer.strokeColor = UIColor.systemYellow.cgColor
        ellipsesLayer.lineWidth = 2
        ellipsesLayer.contentsScale = UIScreen.main.scale
        overlayLayer.addSublayer(ellipsesLayer)

        poseLinesLayer.strokeColor = UIColor.green.cgColor
        poseLinesLayer.fillColor = UIColor.clear.cgColor
        poseLinesLayer.lineWidth = 2.5
        poseLinesLayer.contentsScale = UIScreen.main.scale
        overlayLayer.addSublayer(poseLinesLayer)

        poseCirclesLayer.fillColor = UIColor.yellow.cgColor // 统一颜色，减少多层创建
        poseCirclesLayer.strokeColor = UIColor.white.cgColor
        poseCirclesLayer.lineWidth = 1.5
        poseCirclesLayer.contentsScale = UIScreen.main.scale
        overlayLayer.addSublayer(poseCirclesLayer)

        setupPipelines()
        configureSession()
    }

    func setupPipelines() {
        // 尝试加载bundle中的模型；请将模型添加到Xcode工程的资源中
        objectPipeline = try? ObjectDetectPipeline(modelName: "yolo11n")
        posePipeline = try? PoseDetectPipeline(modelName: "yolon-pose")
        if posePipeline == nil {
            // 使用Apple内置人体姿态作为后备
            applePoseRequest = VNDetectHumanBodyPoseRequest()
            statusText = "未找到自定义姿态模型，使用Apple人体姿态"
            print("[Camera] Pose model missing, using Apple VNDetectHumanBodyPoseRequest fallback")
        } else {
            statusText = "模型加载成功"
        }
    }

    func configureSession() {
        session.beginConfiguration()
        if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        } else {
            session.sessionPreset = .high
        }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        guard let input = try? AVCaptureDeviceInput(device: device) else { return }
        if session.canAddInput(input) { session.addInput(input) }
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        if let conn = videoOutput.connection(with: .video) {
            conn.videoOrientation = .portrait
        }
        session.commitConfiguration()
    }

    private func ensureAuthorization(_ completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            completion(false)
        }
    }
    func startSession() {
        ensureAuthorization { granted in
            guard granted else { self.statusText = "相机未授权"; return }
            guard !self.session.inputs.isEmpty, !self.session.outputs.isEmpty else { self.statusText = "相机会话未就绪"; return }
            if !self.session.isRunning {
                DispatchQueue.global(qos: .userInitiated).async {
                    self.session.startRunning()
                }
            }
        }
    }
    func stopSession() { if session.isRunning { session.stopRunning() } }

    private var lastInferenceTS: CFAbsoluteTime = 0
    var targetInferenceFPS: Double = 20 // 可调整的推理帧率上限

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CFAbsoluteTimeGetCurrent()
        let interval = 1.0 / targetInferenceFPS
        if now - lastInferenceTS < interval {
            return // 节流推理，降低每帧开销
        }
        lastInferenceTS = now
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        switch mode {
        case .objects:
            if let results = try? objectPipeline?.detectObjects(in: pixelBuffer, minConfidence: confidence) {
                DispatchQueue.main.async {
                    let frameSize = self.overlayLayer.bounds.size
                    CATransaction.begin(); CATransaction.setDisableActions(true)
                    self.ellipsesLayer.frame = CGRect(origin: .zero, size: frameSize)
                    self.ellipsesLayer.path = OverlayRenderer.makeEllipsesPath(boxes: results, frameSize: frameSize)
                    // 清空姿态层
                    self.poseLinesLayer.path = nil
                    self.poseCirclesLayer.path = nil
                    CATransaction.commit()
                    self.statusText = results.isEmpty ? "未检测到目标" : "检测到目标：\(results.count)"
                }
            } else {
                DispatchQueue.main.async { self.statusText = "对象模型未加载" }
            }
        case .pose:
            if let poses = try? posePipeline?.predictPoses(in: pixelBuffer, confidence: confidence, kpThreshold: kpThreshold) {
                DispatchQueue.main.async {
                    let frameSize = self.overlayLayer.bounds.size
                    CATransaction.begin(); CATransaction.setDisableActions(true)
                    self.ellipsesLayer.path = nil
                    self.poseLinesLayer.frame = CGRect(origin: .zero, size: frameSize)
                    self.poseCirclesLayer.frame = CGRect(origin: .zero, size: frameSize)
                    let paths = OverlayRenderer.makePosePaths(poses: poses, frameSize: frameSize)
                    self.poseLinesLayer.path = paths.lines
                    self.poseCirclesLayer.path = paths.circles
                    CATransaction.commit()
                    self.statusText = poses.isEmpty ? "未检测到人" : "检测到人：\(poses.count)"
                }
            } else if let req = applePoseRequest {
                // Fallback using Apple VNDetectHumanBodyPoseRequest
                var poses: [Pose] = []
                do {
                    try visionSequence.perform([req], on: pixelBuffer)
                    if let obs = req.results as? [VNHumanBodyPoseObservation] {
                        let frameSize = self.overlayLayer.bounds.size
                        let w = frameSize.width
                        let h = frameSize.height
                        let sFitPre = min(640.0 / w, 640.0 / h)
                        let padX = (640.0 - w * sFitPre) / 2.0
                        let padY = (640.0 - h * sFitPre) / 2.0
                        for o in obs {
                            guard let points = try? o.recognizedPoints(.all) else { continue }
                            // COCO17顺序的映射
                            let order: [VNHumanBodyPoseObservation.JointName] = [
                                .nose, .leftEye, .rightEye, .leftEar, .rightEar,
                                .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
                                .leftWrist, .rightWrist, .leftHip, .rightHip,
                                .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
                            ]
                            var kps: [Keypoint] = []
                            var xs: [CGFloat] = []
                            var ys: [CGFloat] = []
                            for j in order {
                                if let p = points[j], p.confidence >= kpThreshold {
                                    // 转换到原图坐标，再映射到640 letterbox坐标
                                    let xOrig = CGFloat(p.x) * w
                                    let yOrig = (1.0 - CGFloat(p.y)) * h
                                    let mx = padX + xOrig * sFitPre
                                    let my = padY + yOrig * sFitPre
                                    kps.append(Keypoint(x: mx, y: my, conf: p.confidence))
                                    xs.append(mx); ys.append(my)
                                } else {
                                    // 占位，降低valid计数
                                    kps.append(Keypoint(x: 0, y: 0, conf: 0))
                                }
                            }
                            let validCount = kps.reduce(0) { $1.conf >= kpThreshold ? $0 + 1 : $0 }
                            if validCount < 6 { continue }
                            let minX = xs.min() ?? 0
                            let maxX = xs.max() ?? 0
                            let minY = ys.min() ?? 0
                            let maxY = ys.max() ?? 0
                            let bbox = CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))
                            poses.append(Pose(bbox: bbox, score: Float(validCount) / 17.0, keypoints: kps))
                        }
                    }
                } catch {
                    print("[Camera] Apple pose request failed: \(error)")
                }
                DispatchQueue.main.async {
                    let frameSize = self.overlayLayer.bounds.size
                    CATransaction.begin(); CATransaction.setDisableActions(true)
                    self.ellipsesLayer.path = nil
                    self.poseLinesLayer.frame = CGRect(origin: .zero, size: frameSize)
                    self.poseCirclesLayer.frame = CGRect(origin: .zero, size: frameSize)
                    let paths = OverlayRenderer.makePosePaths(poses: poses, frameSize: frameSize)
                    self.poseLinesLayer.path = paths.lines
                    self.poseCirclesLayer.path = paths.circles
                    CATransaction.commit()
                    self.statusText = poses.isEmpty ? "未检测到人（Apple后备）" : "检测到人：\(poses.count)（Apple后备）"
                }
            } else {
                DispatchQueue.main.async { self.statusText = "姿态模型未加载" }
            }
        }
    }
}