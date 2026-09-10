import Foundation
import simd

/// The phone-to-PC wire format.
///
/// Every message is `magic(4) type(1) length(2 LE)` followed by `length` bytes.
/// Framing matters here in a way it did not over UDP: TCP is a byte stream, so
/// without an explicit envelope a single dropped byte would silently reinterpret
/// every landmark that follows it as garbage rather than losing one frame.
///
/// The reverse direction (PC to phone) is newline-delimited JSON instead. Mixing
/// the two is deliberate: the downstream path is 120 packets a second and wants
/// to be cheap and fixed-size, while the upstream path is a handful of settings
/// changes a minute and wants to be readable and extensible.
enum Wire {
    static let magic: [UInt8] = Array("HPU1".utf8)
    static let landmarkCount = 21

    static let typeFrame: UInt8 = 1
    static let typeStatus: UInt8 = 2
    static let typePong: UInt8 = 3
}

/// One frame's worth of hand data.
///
/// Coordinates are normalised 0...1 against the *oriented* image, origin
/// top-left, y downwards -- i.e. already rotated into the frame the user sees,
/// and already flipped out of Vision's bottom-left convention. Doing both on the
/// phone means the desktop never has to know which way the phone was mounted in
/// order to interpret a coordinate.
struct HandFrame {
    var sequence: UInt32
    var captureTime: Double        // sensor presentation timestamp, phone clock
    var sendTime: Double           // just before handing to the socket
    var handPresent: Bool
    var chirality: UInt8           // 0 unknown, 1 left, 2 right
    var orientation: UInt8         // 0 portrait, 1 landLeft, 2 landRight, 3 upsideDown
    var width: UInt16              // oriented dimensions, so aspect is correct
    var height: UInt16
    var inferenceMs: Float
    var points: [SIMD3<Float>]     // x, y, confidence
}

enum WireEncoder {

    static func envelope(type: UInt8, payload: Data) -> Data {
        var out = Data(capacity: 7 + payload.count)
        out.append(contentsOf: Wire.magic)
        out.append(type)
        appendLE(&out, UInt16(payload.count))
        out.append(payload)
        return out
    }

    static func frame(_ f: HandFrame) -> Data {
        var p = Data(capacity: 32 + Wire.landmarkCount * 12)
        appendLE(&p, f.sequence)
        appendLE(&p, f.captureTime.bitPattern)
        appendLE(&p, f.sendTime.bitPattern)
        p.append(f.handPresent ? 1 : 0)
        p.append(f.chirality)
        p.append(f.orientation)
        p.append(0)                                  // reserved, keeps 4-byte tidiness
        appendLE(&p, f.width)
        appendLE(&p, f.height)
        appendLE(&p, f.inferenceMs.bitPattern)
        for i in 0..<Wire.landmarkCount {
            let v = i < f.points.count ? f.points[i] : SIMD3<Float>(0, 0, 0)
            appendLE(&p, v.x.bitPattern)
            appendLE(&p, v.y.bitPattern)
            appendLE(&p, v.z.bitPattern)
        }
        return envelope(type: Wire.typeFrame, payload: p)
    }

    static func json(type: UInt8, _ object: [String: Any]) -> Data {
        let body = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
        return envelope(type: type, payload: body)
    }

    private static func appendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
}
