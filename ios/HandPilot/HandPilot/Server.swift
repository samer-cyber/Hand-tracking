import Foundation
import Network

/// A TCP server that the desktop connects *into*.
///
/// The direction is the whole trick behind running this over the Lightning/USB-C
/// cable. Apple's USB multiplexer (the thing iTunes installs, and what Xcode and
/// `iproxy` use) can open a tunnel from the PC to a TCP port *inside* the phone,
/// but there is no route the other way. So the phone listens and the PC dials in.
///
/// The same listener also serves plain Wi-Fi with no code change -- over Wi-Fi
/// the PC just dials the phone's IP instead of going through the mux. USB is the
/// default because it removes the radio's several-millisecond, highly variable
/// contribution to a budget where the whole point is single-digit milliseconds.
final class ControlServer: ObservableObject {

    @Published var listening = false
    @Published var connected = false
    @Published var lastError = ""
    @Published var framesSent = 0

    /// Called on the network queue when the desktop sends a settings command.
    var onCommand: (([String: Any]) -> Void)?

    private var listener: NWListener?
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "handpilot.net", qos: .userInteractive)
    private var rxBuffer = Data()

    /// Sends outstanding on the socket. If USB stalls for a few milliseconds at
    /// 120 packets a second, an unbounded queue would build a backlog of hand
    /// positions that are already wrong by the time they arrive. Dropping is the
    /// correct response to congestion for live pointing data -- a newer frame has
    /// already superseded whatever is waiting.
    private var inFlight = 0
    private let inFlightLimit = 6
    private var dropped = 0

    func start(port: UInt16) {
        queue.async { [weak self] in self?.beginListening(port: port) }
    }

    private func beginListening(port: UInt16) {
        listener?.cancel()
        connection?.cancel()
        connection = nil

        let tcp = NWProtocolTCP.Options()
        // Nagle batches small writes to save packets. Our packets are small *and*
        // urgent, which is exactly the case Nagle handles badly -- it would add up
        // to 40 ms waiting for a companion packet that never comes.
        tcp.noDelay = true
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 2
        tcp.keepaliveInterval = 2
        tcp.keepaliveCount = 3
        let params = NWParameters(tls: nil, tcp: tcp)
        params.allowLocalEndpointReuse = true

        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            publish { self.lastError = "bad port"; self.listening = false }
            return
        }
        do {
            let l = try NWListener(using: params, on: nwPort)
            l.newConnectionHandler = { [weak self] conn in self?.accept(conn) }
            l.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    self.publish { self.listening = true; self.lastError = "" }
                case .failed(let e):
                    self.publish { self.listening = false; self.lastError = e.localizedDescription }
                case .cancelled:
                    self.publish { self.listening = false }
                default:
                    break
                }
            }
            l.start(queue: queue)
            listener = l
        } catch {
            publish { self.lastError = error.localizedDescription; self.listening = false }
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.connection?.cancel()
            self.connection = nil
            self.listener?.cancel()
            self.listener = nil
            self.publish { self.connected = false; self.listening = false }
        }
    }

    private func accept(_ conn: NWConnection) {
        // One desktop at a time. A second connection replaces the first rather
        // than being refused, so a crashed-and-restarted PC app reconnects
        // cleanly instead of being locked out by its own dead socket.
        connection?.cancel()
        connection = conn
        rxBuffer.removeAll(keepingCapacity: true)
        inFlight = 0

        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.publish { self.connected = true; self.lastError = "" }
                self.receiveLoop(conn)
            case .failed(let e):
                self.publish { self.connected = false; self.lastError = e.localizedDescription }
            case .cancelled:
                self.publish { self.connected = false }
            default:
                break
            }
        }
        conn.start(queue: queue)
    }

    private func receiveLoop(_ conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let data, !data.isEmpty {
                self.rxBuffer.append(data)
                self.drainCommands()
            }
            if isComplete || error != nil {
                conn.cancel()
                self.publish { self.connected = false }
                return
            }
            self.receiveLoop(conn)
        }
    }

    private func drainCommands() {
        while let nl = rxBuffer.firstIndex(of: 0x0A) {
            let line = rxBuffer[rxBuffer.startIndex..<nl]
            rxBuffer.removeSubrange(rxBuffer.startIndex...nl)
            guard !line.isEmpty,
                  let obj = try? JSONSerialization.jsonObject(with: Data(line)),
                  let dict = obj as? [String: Any] else { continue }
            onCommand?(dict)
        }
        // A peer that never sends a newline must not be able to grow this
        // without limit.
        if rxBuffer.count > 64_000 { rxBuffer.removeAll(keepingCapacity: true) }
    }

    /// Fire-and-forget send, safe to call from the capture queue.
    func send(_ data: Data, droppable: Bool) {
        queue.async { [weak self] in
            guard let self, let conn = self.connection, conn.state == .ready else { return }
            if droppable && self.inFlight >= self.inFlightLimit {
                self.dropped += 1
                return
            }
            self.inFlight += 1
            conn.send(content: data, completion: .contentProcessed { [weak self] error in
                guard let self else { return }
                self.inFlight = max(0, self.inFlight - 1)
                if let error {
                    self.publish { self.lastError = error.localizedDescription }
                } else if droppable {
                    self.framesSentTick += 1
                    if self.framesSentTick >= 30 {
                        let n = self.framesSentTick
                        self.framesSentTick = 0
                        self.publish { self.framesSent += n }
                    }
                }
            })
        }
    }

    /// SwiftUI republishes the whole view on every change. At 120 Hz that alone
    /// would cost more than the tracking does, so the counter is batched.
    private var framesSentTick = 0

    var droppedCount: Int { dropped }

    private func publish(_ block: @escaping () -> Void) {
        DispatchQueue.main.async(execute: block)
    }
}
