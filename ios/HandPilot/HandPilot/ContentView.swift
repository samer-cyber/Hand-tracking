import AVFoundation
import SwiftUI

/// The camera settings, persisted, and mirrored to the engine whenever they
/// change from either side.
///
/// Both sides matter. With the rear camera the phone stands pointing at you
/// with its screen facing away, so the computer is the only interface you can
/// see -- but the phone's own controls are what you use to get it aimed and
/// working in the first place.
final class SettingsStore: ObservableObject {
    @Published var port: String { didSet { UserDefaults.standard.set(port, forKey: "port") } }
    @Published var settings: CameraSettings { didSet { persist() } }

    init() {
        let d = UserDefaults.standard
        port = d.string(forKey: "port") ?? "9871"
        var s = CameraSettings()
        if d.object(forKey: "fps") != nil { s.fps = d.double(forKey: "fps") }
        if d.object(forKey: "shutter") != nil { s.shutterDenominator = d.double(forKey: "shutter") }
        if d.object(forKey: "isoAuto") != nil { s.isoAuto = d.bool(forKey: "isoAuto") }
        if d.object(forKey: "isoCeiling") != nil { s.isoCeiling = d.float(forKey: "isoCeiling") }
        if d.object(forKey: "zoom") != nil { s.zoom = d.float(forKey: "zoom") }
        if d.object(forKey: "focusAuto") != nil { s.focusAuto = d.bool(forKey: "focusAuto") }
        if d.object(forKey: "lens") != nil { s.lensPosition = d.float(forKey: "lens") }
        if d.object(forKey: "orient") != nil { s.orientationOverride = d.integer(forKey: "orient") }
        if d.object(forKey: "front") != nil { s.useFrontCamera = d.bool(forKey: "front") }
        if d.object(forKey: "crop") != nil { s.cropToHand = d.bool(forKey: "crop") }
        if d.object(forKey: "preview") != nil { s.showPreview = d.bool(forKey: "preview") }
        settings = s
    }

    private func persist() {
        let d = UserDefaults.standard
        d.set(settings.fps, forKey: "fps")
        d.set(settings.shutterDenominator, forKey: "shutter")
        d.set(settings.isoAuto, forKey: "isoAuto")
        d.set(settings.isoCeiling, forKey: "isoCeiling")
        d.set(settings.zoom, forKey: "zoom")
        d.set(settings.focusAuto, forKey: "focusAuto")
        d.set(settings.lensPosition, forKey: "lens")
        d.set(settings.orientationOverride, forKey: "orient")
        d.set(settings.useFrontCamera, forKey: "front")
        d.set(settings.cropToHand, forKey: "crop")
        d.set(settings.showPreview, forKey: "preview")
    }

    /// Apply a settings command that arrived from the desktop. Unknown keys are
    /// ignored rather than rejected, so an older phone build stays usable against
    /// a newer desktop build.
    func applyCommand(_ c: [String: Any]) {
        var s = settings
        if let v = c["fps"] as? Double { s.fps = v }
        if let v = c["shutter"] as? Double { s.shutterDenominator = v }
        if let v = c["iso_auto"] as? Bool { s.isoAuto = v }
        if let v = c["iso"] as? Double { s.manualISO = Float(v) }
        if let v = c["iso_ceiling"] as? Double { s.isoCeiling = Float(v) }
        if let v = c["zoom"] as? Double { s.zoom = Float(v) }
        if let v = c["focus_auto"] as? Bool { s.focusAuto = v }
        if let v = c["lens"] as? Double { s.lensPosition = Float(v) }
        if let v = c["orientation"] as? Int { s.orientationOverride = v }
        if let v = c["camera"] as? String { s.useFrontCamera = (v == "front") }
        if let v = c["crop"] as? Bool { s.cropToHand = v }
        if let v = c["preview"] as? Bool { s.showPreview = v }
        settings = s
    }
}

struct ContentView: View {
    @StateObject private var engine = CaptureEngine()
    @StateObject private var server = ControlServer()
    @StateObject private var store = SettingsStore()
    @State private var running = false

    private let shutterChoices: [Double] = [120, 250, 300, 360, 500, 750, 1000]
    private let fpsChoices: [Double] = [60, 120, 240]

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                header
                handPanel
                startButton
                stats
                cameraCard
                Text("Stand the phone a forearm's length away, camera on you. Everything here can also "
                     + "be changed from HandPilot on your computer.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 8)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 20)
            .frame(maxWidth: 620)
            .frame(maxWidth: .infinity)
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear {
            // Bound to locals first: these are the objects the escaping command
            // handler needs to outlive this call, and capturing them by name
            // keeps the View struct itself out of the closure.
            let net = server
            let cfg = store
            engine.attach(server: net)
            net.onCommand = { [weak net, weak cfg] command in
                if let ping = command["ping"] {
                    // Echoed straight back so the desktop can measure the round
                    // trip through the cable without a synchronised clock.
                    net?.send(WireEncoder.json(type: Wire.typePong, ["ping": ping]),
                              droppable: false)
                    return
                }
                DispatchQueue.main.async { cfg?.applyCommand(command) }
            }
        }
        .onChange(of: store.settings) { _, new in engine.apply(new) }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("HandPilot").font(.title.bold())
                Text(connectionText)
                    .font(.subheadline)
                    .foregroundStyle(connectionColor)
            }
            Spacer()
            Circle()
                .fill(connectionColor)
                .frame(width: 12, height: 12)
                .accessibilityHidden(true)
        }
    }

    private var connectionText: String {
        if server.connected { return "Computer connected" }
        if server.listening { return "Waiting for your computer" }
        if running { return "Starting…" }
        return "Stopped"
    }

    private var connectionColor: Color {
        if server.connected { return .green }
        if server.listening { return .orange }
        return .secondary
    }

    private var handPanel: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18).fill(Color.white.opacity(0.05))
            if running && store.settings.showPreview {
                CameraPreview(session: engine.session, device: engine.activeDevice)
                    .clipShape(RoundedRectangle(cornerRadius: 18))
            } else {
                SkeletonView(live: engine.liveHand, mirrored: store.settings.useFrontCamera)
                    .padding(12)
            }
            if !running {
                VStack(spacing: 8) {
                    Image(systemName: "hand.raised")
                        .font(.system(size: 34))
                    Text("Tap Start, then point the camera at yourself")
                        .font(.callout)
                }
                .foregroundStyle(.secondary)
            } else if !engine.handDetected && !store.settings.showPreview {
                Text("No hand in view")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(height: 260)
        .overlay(RoundedRectangle(cornerRadius: 18)
            .stroke(engine.handDetected ? Color.green.opacity(0.85) : Color.white.opacity(0.12),
                    lineWidth: engine.handDetected ? 2 : 1))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(engine.handDetected ? "Hand detected" : "No hand detected")
    }

    private var startButton: some View {
        Button(action: toggleRunning) {
            Label(running ? "Stop" : "Start", systemImage: running ? "stop.fill" : "play.fill")
                .font(.title3.bold())
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
        }
        .buttonStyle(.borderedProminent)
        .tint(running ? .red : .green)
        .controlSize(.large)
    }

    private var stats: some View {
        let columns = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10),
                       GridItem(.flexible(), spacing: 10)]
        return LazyVGrid(columns: columns, spacing: 10) {
            Stat(title: "Tracking", value: String(format: "%.0f fps", engine.trackedFPS),
                 tint: engine.captureFPS > 0 && engine.trackedFPS < engine.captureFPS * 0.8 ? .orange : .primary)
            Stat(title: "Hand model", value: String(format: "%.1f ms", engine.inferenceMs), tint: .primary)
            Stat(title: "Hand", value: engine.handDetected ? "Tracking" : "None",
                 tint: engine.handDetected ? .green : .secondary)
            Stat(title: "ISO", value: String(format: "%.0f", engine.isoValue),
                 tint: abs(engine.exposureOffset) < 1.0 ? .primary : .orange)
            Stat(title: "Format", value: engine.formatSummary, tint: .primary)
            Stat(title: "Camera", value: engine.status, tint: .secondary)
        }
        .overlay(alignment: .bottom) {
            if !server.lastError.isEmpty {
                Text(server.lastError)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .offset(y: 18)
            }
        }
    }

    private var cameraCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker("Camera", selection: binding(\.useFrontCamera)) {
                Text("Back camera").tag(false)
                Text("Front camera").tag(true)
            }
            .pickerStyle(.segmented)

            Picker("Phone screen shows", selection: binding(\.showPreview)) {
                Text("Camera").tag(true)
                Text("Skeleton").tag(false)
            }
            .pickerStyle(.segmented)

            LabeledContent("Frame rate") {
                Picker("Frame rate", selection: binding(\.fps)) {
                    ForEach(fpsChoices, id: \.self) { Text("\(Int($0))").tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 190)
            }

            LabeledContent("Shutter") {
                Picker("Shutter", selection: binding(\.shutterDenominator)) {
                    ForEach(shutterChoices, id: \.self) { Text("1/\(Int($0)) s").tag($0) }
                }
            }

            Toggle("Hand crop (sharper landmarks)", isOn: binding(\.cropToHand))

            DisclosureGroup("More camera settings") {
                VStack(alignment: .leading, spacing: 14) {
                    LabeledContent("Orientation") {
                        Picker("Orientation", selection: binding(\.orientationOverride)) {
                            Text("Automatic").tag(-1)
                            Text("Portrait").tag(0)
                            Text("Landscape left").tag(1)
                            Text("Landscape right").tag(2)
                            Text("Upside down").tag(3)
                        }
                    }
                    Toggle("Automatic ISO", isOn: binding(\.isoAuto))
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Zoom  \(store.settings.zoom, specifier: "%.1f")×")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Slider(value: zoomBinding, in: 1...3, step: 0.1)
                    }
                    Toggle("Autofocus", isOn: binding(\.focusAuto))
                    if !store.settings.focusAuto {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Focus  \(store.settings.lensPosition, specifier: "%.2f")")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Slider(value: lensBinding, in: 0...1)
                        }
                    }
                    LabeledContent("Port") {
                        TextField("9871", text: $store.port)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 90)
                    }
                }
                .padding(.top, 10)
            }
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color.white.opacity(0.06)))
    }

    // MARK: - Helpers

    private func toggleRunning() {
        running.toggle()
        if running {
            server.start(port: UInt16(store.port) ?? 9871)
            engine.apply(store.settings)
            engine.start()
        } else {
            engine.stop()
            server.stop()
        }
    }

    private func binding<Value>(_ keyPath: WritableKeyPath<CameraSettings, Value>) -> Binding<Value> {
        Binding(get: { store.settings[keyPath: keyPath] },
                set: { store.settings[keyPath: keyPath] = $0 })
    }

    private var zoomBinding: Binding<Double> {
        Binding(get: { Double(store.settings.zoom) }, set: { store.settings.zoom = Float($0) })
    }

    private var lensBinding: Binding<Double> {
        Binding(get: { Double(store.settings.lensPosition) },
                set: { store.settings.lensPosition = Float($0) })
    }
}

/// One live number, small enough that six fit on a phone.
private struct Stat: View {
    let title: String
    let value: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .foregroundStyle(tint)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.06)))
        .accessibilityElement(children: .combine)
    }
}

/// The tracked hand, drawn from the same landmarks the computer receives --
/// so what you see here is exactly what the pointer is riding on.
struct SkeletonView: View {
    @ObservedObject var live: LiveHand
    /// Mirror the drawing so the front camera reads like a mirror.
    var mirrored: Bool

    private static let bones: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
        (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
        (17, 18), (18, 19), (19, 20), (0, 17),
    ]
    private static let tips: Set<Int> = [4, 8, 12, 16, 20]

    var body: some View {
        Canvas { context, size in
            let snap = live.snapshot
            var width = size.width
            var height = width / max(snap.aspect, 0.1)
            if height > size.height {
                height = size.height
                width = height * snap.aspect
            }
            let origin = CGPoint(x: (size.width - width) / 2, y: (size.height - height) / 2)
            let frame = CGRect(origin: origin, size: CGSize(width: width, height: height))
            context.stroke(Path(roundedRect: frame, cornerRadius: 10),
                           with: .color(.white.opacity(0.12)), lineWidth: 1)

            guard snap.present, snap.points.count == 21, snap.visible.count == 21 else { return }

            func place(_ p: CGPoint) -> CGPoint {
                let x = mirrored ? 1 - p.x : p.x
                return CGPoint(x: origin.x + x * width, y: origin.y + p.y * height)
            }

            var bones = Path()
            for (a, b) in Self.bones where snap.visible[a] && snap.visible[b] {
                bones.move(to: place(snap.points[a]))
                bones.addLine(to: place(snap.points[b]))
            }
            context.stroke(bones, with: .color(.green),
                           style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))

            for i in 0..<21 where snap.visible[i] {
                let c = place(snap.points[i])
                let r: CGFloat = Self.tips.contains(i) ? 4.5 : 3
                let dot = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2))
                context.fill(dot, with: .color(Self.tips.contains(i) ? .white : .green))
            }
        }
    }
}

/// A plain preview layer, so you can aim the phone. It is a choice rather than
/// a fixture because at 1080p120 the extra display path is real power and real
/// heat on a phone already running the neural engine flat out.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let device: AVCaptureDevice?

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        view.follow(device)
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.follow(device)
    }

    /// Keeps the picture upright as the phone turns, for whichever camera is
    /// live: the rotation coordinator knows how each sensor is mounted.
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        private var coordinator: AVCaptureDevice.RotationCoordinator?
        private var observation: NSKeyValueObservation?
        private weak var followed: AVCaptureDevice?

        func follow(_ device: AVCaptureDevice?) {
            guard device !== followed else { return }
            followed = device
            observation = nil
            coordinator = nil
            guard let device else { return }
            let c = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: videoPreviewLayer)
            coordinator = c
            rotate(to: c.videoRotationAngleForHorizonLevelPreview)
            observation = c.observe(\.videoRotationAngleForHorizonLevelPreview,
                                    options: [.new]) { [weak self] observed, _ in
                let angle = observed.videoRotationAngleForHorizonLevelPreview
                DispatchQueue.main.async { self?.rotate(to: angle) }
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            // The layer's connection only exists once the session has a camera,
            // which can be after `follow` ran.
            if let c = coordinator { rotate(to: c.videoRotationAngleForHorizonLevelPreview) }
        }

        private func rotate(to angle: CGFloat) {
            guard let connection = videoPreviewLayer.connection,
                  connection.isVideoRotationAngleSupported(angle) else { return }
            connection.videoRotationAngle = angle
        }
    }
}
