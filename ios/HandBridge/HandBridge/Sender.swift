import Foundation
import Network
import QuartzCore
import simd

/// One frame's worth of hand data, as it goes onto the wire.
struct HandPacket {
    let sequence: UInt32
    let captureTime: Double       // phone clock, seconds, taken at frame arrival
    let sendTime: Double          // phone clock, seconds, just before transmit
    let width: UInt16
    let height: UInt16
    let intrinsics: SIMD4<Float>  // fx, fy, cx, cy for the raw buffer
    let chirality: UInt8          // 0 unknown, 1 left, 2 right
    let landmarks: [SIMD2<Float>]?
    let confidence: [Float]?
    let depth: [Float]?
}

/// Sends hand packets to the desktop over UDP.
///
/// UDP rather than TCP deliberately. TCP guarantees delivery by retransmitting,
/// which means one lost packet stalls everything behind it -- and a hand
/// position that arrives late is worthless, because a newer one has already
/// replaced it. Dropping a frame outright is the correct behaviour here, and
/// that is exactly what UDP does for free.
final class PacketSender: ObservableObject {

    @Published var isConnected = false
    @Published var lastError: String = ""
    @Published var packetsSent: Int = 0

    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "handbridge.net", qos: .userInteractive)
    private static let magic: [UInt8] = Array("HBR1".utf8)
    private static let landmarkCount = 21
    /// 4 magic + 4 seq + 8 capture + 8 send + 2 w + 2 h + 16 intrinsics
    /// + 1 chirality + 1 hasHand + 1 hasDepth + 1 pad + 21*16 landmarks
    static let packetSize = 48 + landmarkCount * 16

    func connect(host: String, port: UInt16) {
        disconnect()
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            DispatchQueue.main.async { self.lastError = "bad port" }
            return
        }
        let conn = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: .udp)
        conn.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                switch state {
                case .ready:
                    self?.isConnected = true
                    self?.lastError = ""
                case .failed(let err):
                    self?.isConnected = false
                    self?.lastError = err.localizedDescription
                case .waiting(let err):
                    self?.isConnected = false
                    self?.lastError = "waiting: \(err.localizedDescription)"
                case .cancelled:
                    self?.isConnected = false
                default:
                    break
                }
            }
        }
        connection = conn
        conn.start(queue: queue)
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
        DispatchQueue.main.async { self.isConnected = false }
    }

    func send(_ packet: HandPacket) {
        guard let connection, connection.state == .ready else { return }
        let data = encode(packet)
        connection.send(content: data, completion: .contentProcessed { [weak self] error in
            if let error {
                DispatchQueue.main.async { self?.lastError = error.localizedDescription }
            } else {
                DispatchQueue.main.async { self?.packetsSent += 1 }
            }
        })
    }

    private func encode(_ p: HandPacket) -> Data {
        var data = Data(capacity: Self.packetSize)
        data.append(contentsOf: Self.magic)
        appendLE(&data, p.sequence)
        appendLE(&data, p.captureTime.bitPattern)
        appendLE(&data, p.sendTime.bitPattern)
        appendLE(&data, p.width)
        appendLE(&data, p.height)
        for v in [p.intrinsics.x, p.intrinsics.y, p.intrinsics.z, p.intrinsics.w] {
            appendLE(&data, v.bitPattern)
        }
        data.append(p.chirality)
        data.append(p.landmarks != nil ? 1 : 0)
        data.append(p.depth != nil ? 1 : 0)
        data.append(0)   // padding, keeps the landmark block 4-byte aligned

        for i in 0..<Self.landmarkCount {
            let pt = p.landmarks?[i] ?? SIMD2<Float>(.nan, .nan)
            let z = p.depth?[i] ?? .nan
            let c = p.confidence?[i] ?? 0
            appendLE(&data, pt.x.bitPattern)
            appendLE(&data, pt.y.bitPattern)
            appendLE(&data, z.bitPattern)
            appendLE(&data, c.bitPattern)
        }
        return data
    }

    private func appendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
}
