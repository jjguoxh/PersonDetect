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

    override init() {
        super.init()
        overlayLayer.frame = .zero
        overlayLayer.masksToBounds = true
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
        session.sessionPreset = .high
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

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        switch mode {
        case .objects:
            if let results = try? objectPipeline?.detectObjects(in: pixelBuffer, minConfidence: confidence) {
                DispatchQueue.main.async {
                    let frameSize = self.overlayLayer.bounds.size
                    self.overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
                    OverlayRenderer.drawEllipses(on: self.overlayLayer, boxes: results, frameSize: frameSize)
                }
            }
        case .pose:
            if let poses = try? posePipeline?.predictPoses(in: pixelBuffer, confidence: confidence, kpThreshold: kpThreshold) {
                DispatchQueue.main.async {
                    let frameSize = self.overlayLayer.bounds.size
                    self.overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
                    OverlayRenderer.drawPoses(on: self.overlayLayer, poses: poses, frameSize: frameSize)
                }
            }
        }
    }
}