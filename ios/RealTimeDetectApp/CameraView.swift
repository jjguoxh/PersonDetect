import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var controller = CameraViewController()
    @State private var mode: DetectMode = .objects
    @State private var confidence: Float = 0.5
    @State private var kpThreshold: Float = 0.3

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                CameraPreviewLayer(session: controller.session)
                    .onAppear { controller.startSession() }
                    .onDisappear { controller.stopSession() }
                OverlayContainer(layer: controller.overlayLayer)
            }
            .background(Color.black)
            .ignoresSafeArea() // 让预览层与覆盖层在 iPhone/iPad 上都填满屏幕安全区域

            HStack {
                Picker("模式", selection: $mode) {
                    Text("椭圆").tag(DetectMode.objects)
                    Text("关节").tag(DetectMode.pose)
                }
                .pickerStyle(.segmented)
                .onChange(of: mode) { newValue in controller.mode = newValue }

                Spacer()

                VStack(alignment: .leading) {
                    HStack {
                        Text("conf: \(String(format: "%.2f", confidence))")
                        Slider(value: Binding(get: { Double(confidence) }, set: { confidence = Float($0); controller.confidence = confidence }), in: 0.0...1.0)
                    }
                    if mode == .pose {
                        HStack {
                            Text("kp: \(String(format: "%.2f", kpThreshold))")
                            Slider(value: Binding(get: { Double(kpThreshold) }, set: { kpThreshold = Float($0); controller.kpThreshold = kpThreshold }), in: 0.0...1.0)
                        }
                    }
                }
            }
            .padding()
        }
    }
}

enum DetectMode { case objects, pose }

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView { PreviewView(session: session) }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

final class PreviewView: UIView {
    private let previewLayer = AVCaptureVideoPreviewLayer()
    init(session: AVCaptureSession) {
        super.init(frame: .zero)
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspect
        layer.addSublayer(previewLayer)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
}

struct OverlayContainer: UIViewRepresentable {
    let layer: CALayer
    func makeUIView(context: Context) -> UIView { OverlayView(layer: layer) }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

final class OverlayView: UIView {
    let overlayLayer: CALayer
    init(layer: CALayer) {
        self.overlayLayer = layer
        super.init(frame: .zero)
        self.backgroundColor = .clear
        self.layer.addSublayer(overlayLayer)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layoutSubviews() {
        super.layoutSubviews()
        overlayLayer.frame = bounds
    }
}