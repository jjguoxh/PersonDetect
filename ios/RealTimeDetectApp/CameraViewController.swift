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

    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.sample.queue")

    // Pipelines
    private var objectPipeline: ObjectDetectPipeline?
    private var posePipeline: PoseDetectPipeline?

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
        ellipsesLayer.strokeColor = UIColor.yellow.cgColor
        ellipsesLayer.fillColor = UIColor.yellow.withAlphaComponent(0.2).cgColor
        ellipsesLayer.lineWidth = 3
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
        objectPipeline = try? ObjectDetectPipeline(modelName: "yolo11x")
        posePipeline = try? PoseDetectPipeline(modelName: "yolon-pose")
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

    func startSession() { if !session.isRunning { session.startRunning() } }
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
                }
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
                }
            }
        }
    }
}