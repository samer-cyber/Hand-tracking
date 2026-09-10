import AVFoundation
import SwiftUI

/// The camera settings, persisted, and mirrored to the engine whenever they
/// change from either side.
///
/// Both sides matter. In normal use the phone is in a stand pointing at you with
/// its screen facing away, so the desktop is the only interface you can actually
/// see -- but the phone's own controls are what you use to get it aimed and
/// working in the first place, before you turn it around.
final class SettingsStore: ObservableObject {
    @Published var port: String { didSet { UserDefaults.standard.set(port, forKey: "port") } }
    @Published var settings: CameraSettings { didSet { persist() } }
    @Published var showPreview: Bool { didSet { UserDefaults.standard.set(showPreview, forKey: "preview") } }

    init() {
        let d = UserDefaults.standard
        port = d.string(forKey: "port") ?? "9871"
        showPreview = d.object(forKey: "preview") as? Bool ?? true
        var s = CameraSettings()
        if d.object(forKey: "fps") != nil { s.fps = d.double(forKey: "fps") }
        if d.object(forKey: "shutter") != nil { s.shutterDenominator = d.double(forKey: "shutter") }
        if d.object(forKey: "isoAuto") != nil { s.isoAuto = d.bool(forKey: "isoAuto") }
        if d.object(forKey: "isoCeiling") != nil { s.isoCeiling = d.float(forKey: "isoCeiling") }
        if d.object(forKey: "zoom") != nil { s.zoom = d.float(forKey: "zoom") }
        if d.object(forKey: "focusAuto") != nil { s.focusAuto = d.bool(forKey: "focusAuto") }
        if d.object(forKey: "lens") != nil { s.lensPosition = d.float(forKey: "lens") }
        if d.object(forKey: "orient") != nil { s.orientationOverride = d.integer(forKey: "orient") }
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
        if let v = c["preview"] as? Bool { showPreview = v }
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

                if store.showPreview && running {
                    CameraPreview(session: engine.session)
                        .frame(height: 160)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(RoundedRectangle(cornerRadius: 12)
                            .stroke(engine.handDetected ? Color.green : Color.gray.opacity(0.4),
                                    lineWidth: 2))
                        .padding(.horizontal)
                }

                startButton
                statusGrid
                controls

                Text("Aim the REAR camera at yourself, then turn the phone around. "
                     + "Everything below can also be driven from the desktop app, "
                     + "which is the point -- you cannot see this screen in use.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            .padding(.vertical, 24)
        }
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

    private var header: some View {
        VStack(spacing: 4) {
            Text("HandPilot").font(.largeTitle.bold())
            Text(server.connected ? "desktop connected"
                                  : (server.listening ? "waiting for desktop" : "not listening"))
                .font(.subheadline)
                .foregroundStyle(server.connected ? .green : .secondary)
        }
    }

    private var startButton: some View {
        Button(running ? "Stop" : "Start") {
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
        .font(.title2.bold())
        .frame(maxWidth: .infinity)
        .padding()
        .background(running ? Color.red.opacity(0.85) : Color.accentColor)
        .foregroundStyle(.white)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal)
    }

    private var statusGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
            row("camera", engine.status, .secondary)
            row("format", engine.formatSummary, .secondary)
            row("capture", String(format: "%.0f fps", engine.captureFPS), .secondary)
            row("tracked", String(format: "%.0f fps", engine.trackedFPS),
                engine.trackedFPS > engine.captureFPS * 0.8 ? .green : .orange)
            row("vision", String(format: "%.1f ms", engine.inferenceMs), .secondary)
            row("hand", engine.handDetected ? "tracking" : "none",
                engine.handDetected ? .green : .orange)
            row("iso", String(format: "%.0f  (%+.1f EV)", engine.isoValue, engine.exposureOffset),
                abs(engine.exposureOffset) < 1.0 ? .secondary : .orange)
            row("mount", engine.orientationName, .secondary)
            row("frames", "\(server.framesSent)", .secondary)
        }
        .font(.system(.footnote, design: .monospaced))
        .padding(.horizontal)
        .overlay(alignment: .bottom) {
            if !server.lastError.isEmpty {
                Text(server.lastError).font(.caption2).foregroundStyle(.red)
                    .offset(y: 16)
            }
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            LabeledContent("Port") {
                TextField("9871", text: $store.port)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.numberPad)
                    .frame(width: 110)
            }

            Picker("Frame rate", selection: Binding(
                get: { store.settings.fps },
                set: { store.settings.fps = $0 })) {
                    ForEach(fpsChoices, id: \.self) { Text("\(Int($0)) fps").tag($0) }
                }
                .pickerStyle(.segmented)

            Picker("Shutter", selection: Binding(
                get: { store.settings.shutterDenominator },
                set: { store.settings.shutterDenominator = $0 })) {
                    ForEach(shutterChoices, id: \.self) { Text("1/\(Int($0))").tag($0) }
                }

            Picker("Mount", selection: Binding(
                get: { store.settings.orientationOverride },
                set: { store.settings.orientationOverride = $0 })) {
                    Text("Auto").tag(-1)
                    Text("Portrait").tag(0)
                    Text("Landscape L").tag(1)
                    Text("Landscape R").tag(2)
                }

            Toggle("Auto ISO", isOn: Binding(
                get: { store.settings.isoAuto },
                set: { store.settings.isoAuto = $0 }))

            VStack(alignment: .leading) {
                Text("Zoom  \(store.settings.zoom, specifier: "%.1f")x")
                    .font(.footnote).foregroundStyle(.secondary)
                Slider(value: Binding(get: { Double(store.settings.zoom) },
                                      set: { store.settings.zoom = Float($0) }),
                       in: 1...3, step: 0.1)
            }

            Toggle("Autofocus", isOn: Binding(
                get: { store.settings.focusAuto },
                set: { store.settings.focusAuto = $0 }))

            if !store.settings.focusAuto {
                VStack(alignment: .leading) {
                    Text("Lens  \(store.settings.lensPosition, specifier: "%.2f")")
                        .font(.footnote).foregroundStyle(.secondary)
                    Slider(value: Binding(get: { Double(store.settings.lensPosition) },
                                          set: { store.settings.lensPosition = Float($0) }),
                           in: 0...1)
                }
            }

            Toggle("Show preview (costs power and heat)", isOn: $store.showPreview)
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private func row(_ label: String, _ value: String, _ colour: Color) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).foregroundStyle(colour)
        }
    }
}

/// A plain preview layer, only there so you can aim the phone before turning it
/// around. It is a toggle rather than a fixture because at 1080p120 the extra
/// display path is real power and real heat on a phone already running the
/// neural engine flat out.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
