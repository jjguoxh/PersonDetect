import Foundation
import UIKit
import Vision

enum OverlayRenderer {
    static func drawEllipses(on layer: CALayer, boxes: [VNRecognizedObjectObservation], frameSize: CGSize) {
        let w = frameSize.width
        let h = frameSize.height
        for obs in boxes {
            let b = obs.boundingBox
            // Vision boundingBox 原点在左下角（归一化）。转换到像素坐标系（左上角视觉）。
            let rect = CGRect(
                x: CGFloat(b.origin.x) * w,
                y: (1.0 - CGFloat(b.origin.y) - CGFloat(b.size.height)) * h,
                width: CGFloat(b.size.width) * w,
                height: CGFloat(b.size.height) * h
            )
            // 可选：调试矩形
            // let box = CAShapeLayer(); box.path = UIBezierPath(rect: rect).cgPath; box.strokeColor = UIColor.red.cgColor; box.fillColor = UIColor.clear.cgColor; box.lineWidth = 2; layer.addSublayer(box)
            let centerX = rect.midX
            let centerY = rect.maxY // 底部锚点
            let ellipseWidth = rect.width / 2.0
            let ellipseHeight = max(8.0, rect.height / 8.0)
            let ellipseRect = CGRect(x: centerX - ellipseWidth / 2.0, y: centerY - ellipseHeight / 2.0, width: ellipseWidth, height: ellipseHeight)
            let e = CAShapeLayer()
            e.path = UIBezierPath(ovalIn: ellipseRect).cgPath
            e.strokeColor = UIColor.yellow.cgColor
            e.fillColor = UIColor.yellow.withAlphaComponent(0.2).cgColor
            e.lineWidth = 3
            layer.addSublayer(e)
        }
    }

    static func drawPoses(on layer: CALayer, poses: [Pose], frameSize: CGSize) {
        let w = frameSize.width
        let h = frameSize.height
        // letterbox 映射（model空间640x640，scaleFit）
        let sFitPre = min(640.0 / w, 640.0 / h)
        let padX = (640.0 - w * sFitPre) / 2.0
        let padY = (640.0 - h * sFitPre) / 2.0
        func mapToOriginal(_ mx: CGFloat, _ my: CGFloat) -> CGPoint {
            let ox = (mx - padX) / sFitPre
            let oy = (my - padY) / sFitPre
            return CGPoint(x: min(max(ox, 0), w-1), y: min(max(oy, 0), h-1))
        }
        let connections: [(Int, Int)] = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,11),(6,12),(11,12),
            (5,7),(7,9),(6,8),(8,10),
            (11,13),(13,15),(12,14),(14,16)
        ]
        for pose in poses {
            // 画关键点
            for (i, kp) in pose.keypoints.enumerated() {
                let p = mapToOriginal(kp.x, kp.y)
                let circle = CAShapeLayer()
                circle.path = UIBezierPath(ovalIn: CGRect(x: p.x-4, y: p.y-4, width: 8, height: 8)).cgPath
                circle.fillColor = (i <= 4 ? UIColor.orange : (i <= 10 ? UIColor.blue : UIColor.yellow)).cgColor
                circle.strokeColor = UIColor.white.cgColor
                circle.lineWidth = 1.5
                layer.addSublayer(circle)
            }
            // 画连线
            for (a, b) in connections {
                let pa = mapToOriginal(pose.keypoints[a].x, pose.keypoints[a].y)
                let pb = mapToOriginal(pose.keypoints[b].x, pose.keypoints[b].y)
                let line = CAShapeLayer()
                let path = UIBezierPath(); path.move(to: pa); path.addLine(to: pb)
                line.path = path.cgPath
                line.strokeColor = UIColor.green.cgColor
                line.lineWidth = 2.5
                layer.addSublayer(line)
            }
        }
    }

    // MARK: - 高效合并路径：GPU友好（少量 CAShapeLayer，更新 path）
    static func makeEllipsesPath(boxes: [VNRecognizedObjectObservation], frameSize: CGSize) -> CGPath {
        let w = frameSize.width
        let h = frameSize.height
        let path = CGMutablePath()
        for obs in boxes {
            let b = obs.boundingBox
            let rect = CGRect(
                x: CGFloat(b.origin.x) * w,
                y: (1.0 - CGFloat(b.origin.y) - CGFloat(b.size.height)) * h,
                width: CGFloat(b.size.width) * w,
                height: CGFloat(b.size.height) * h
            )
            let centerX = rect.midX
            let centerY = rect.maxY
            let ellipseWidth = rect.width / 2.0
            let ellipseHeight = max(8.0, rect.height / 8.0)
            let ellipseRect = CGRect(x: centerX - ellipseWidth / 2.0, y: centerY - ellipseHeight / 2.0, width: ellipseWidth, height: ellipseHeight)
            path.addEllipse(in: ellipseRect)
        }
        return path
    }

    static func makePosePaths(poses: [Pose], frameSize: CGSize) -> (lines: CGPath, circles: CGPath) {
        let w = frameSize.width
        let h = frameSize.height
        let sFitPre = min(640.0 / w, 640.0 / h)
        let padX = (640.0 - w * sFitPre) / 2.0
        let padY = (640.0 - h * sFitPre) / 2.0
        func mapToOriginal(_ mx: CGFloat, _ my: CGFloat) -> CGPoint {
            let ox = (mx - padX) / sFitPre
            let oy = (my - padY) / sFitPre
            return CGPoint(x: min(max(ox, 0), w-1), y: min(max(oy, 0), h-1))
        }
        let connections: [(Int, Int)] = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,11),(6,12),(11,12),
            (5,7),(7,9),(6,8),(8,10),
            (11,13),(13,15),(12,14),(14,16)
        ]
        let lines = CGMutablePath()
        let circles = CGMutablePath()
        for pose in poses {
            for (i, kp) in pose.keypoints.enumerated() {
                let p = mapToOriginal(kp.x, kp.y)
                circles.addEllipse(in: CGRect(x: p.x-4, y: p.y-4, width: 8, height: 8))
            }
            for (a, b) in connections {
                let pa = mapToOriginal(pose.keypoints[a].x, pose.keypoints[a].y)
                let pb = mapToOriginal(pose.keypoints[b].x, pose.keypoints[b].y)
                lines.move(to: pa)
                lines.addLine(to: pb)
            }
        }
        return (lines, circles)
    }
}