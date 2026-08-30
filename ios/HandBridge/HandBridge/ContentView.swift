import SwiftUI

/// Deliberately plain. The only things worth showing are the ones you need to
/// diagnose a session: is it connected, is a hand being seen, how much depth is
/// actually coming back, and at what rate.
struct ContentView: View {
    @StateObject private var engine = CaptureEngine()
    @StateObject private var sender = PacketSender()

    @AppStorage("pcHost") private var host: String = "192.168.1.10"
    @AppStorage("pcPort") private var port: String = "9870"
    @State private var running = false

    var body: some View {
        VStack(spacing: 18) {
            Text("HandBridge")
                .font(.largeTitle.bold())

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("PC address").frame(width: 96, alignment: .leading)
                    TextField("192.168.1.10", text: $host)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.decimalPad)
                        .autocorrectionDisabled()
                }
                HStack {
                    Text("Port").frame(width: 96, alignment: .leading)
                    TextField("9870", text: $port)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.numberPad)
                }
            }
            .padding(.horizontal)

            Button(running ? "Stop" : "Start") {
                running.toggle()
                if running {
                    sender.connect(host: host, port: UInt16(port) ?? 9870)
                    engine.configure(sender: sender)
                    engine.start()
                } else {
                    engine.stop()
                    sender.disconnect()
                }
            }
            .font(.title2.bold())
            .frame(maxWidth: .infinity)
            .padding()
            .background(running ? Color.red.opacity(0.85) : Color.accentColor)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .padding(.horizontal)

            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 8) {
                row("link", sender.isConnected ? "connected" : "not connected",
                    sender.isConnected ? .green : .secondary)
                row("camera", engine.status, .secondary)
                row("rate", String(format: "%.0f fps", engine.framesPerSecond), .secondary)
                row("format", engine.formatSummary, .secondary)
                row("depth align", engine.depthAligned ? "intrinsics" : "SCALED (bad)",
                    engine.depthAligned ? .green : .red)
                row("hand", engine.handDetected ? "tracking" : "none",
                    engine.handDetected ? .green : .orange)
                row("depth", "\(engine.depthCoverage) / 21 landmarks",
                    engine.depthCoverage >= 12 ? .green : .orange)
                row("packets", "\(sender.packetsSent)", .secondary)
            }
            .font(.system(.body, design: .monospaced))
            .padding(.horizontal)

            if !sender.lastError.isEmpty {
                Text(sender.lastError)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Text("Point the front camera at your hands, about 30-60 cm away. "
                 + "TrueDepth returns nothing much past a metre.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()
        }
        .padding(.top, 40)
    }

    @ViewBuilder
    private func row(_ label: String, _ value: String, _ colour: Color) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).foregroundStyle(colour)
        }
    }
}
