import AVFoundation
import CoreVideo
import Foundation
import QuartzCore
import simd
import UIKit
import Vision

/// Captures the front TrueDepth camera, finds hand landmarks on-device, samples
/// depth at each landmark, and hands the result to the sender.
///
/// Three things make this worth more than streaming video to the PC:
///
/// 1. Only ~380 bytes leave the phone per frame instead of a compressed image,
///    so there is no encode, no decode, and no video buffering in the path.
///    Streaming video was the single largest unknown in the latency budget.
/// 2. The depth map is registered to the colour frame, so a landmark found at a
///    pixel can be looked up directly to get its distance in metres.
/// 3. The camera reports its own intrinsic matrix per frame. Every calibration
///    so far has had to *assume* a field of view, and a wrong assumption
///    corrupts the essential-matrix solve that the whole rig geometry rests on.
///    Measured intrinsics remove that error entirely.
final class CaptureEngine: NSObject, ObservableObject {

    @Published var status: String = "idle"
    @Published var framesPerSecond: Double = 0
    @Published var handDetected: Bool = false
    @Published var depthCoverage: Int = 0
    /// What the camera actually negotiated. Worth surfacing: the video and depth
    /// streams can run at different rates, and knowing which is which explains
    /// why depth appears on only some frames.
    @Published var formatSummary: String = "-"
    /// Whether depth is being mapped through real intrinsics rather than by
    /// assuming the two streams share a field of view.
    @Published var depthAligned: Bool = false

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let queue = DispatchQueue(label: "handbridge.capture", qos: .userInteractive)

    private let handRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        // One hand keeps inference cost predictable; every extra hand costs a
        // full pose evaluation and shows up directly as latency.
        r.maximumHandCount = 1
        return r
    }()

    private var sender: PacketSender?
    private var sequence: UInt32 = 0
    /// Physical device orientation, sent with every packet.
    ///
    /// The camera buffer is landscape regardless of how the phone is held, so
    /// landmarks arrive rotated relative to the user's own frame. Rather than
    /// rotating here and risking a mismatch with the intrinsics, the raw frame
    /// is sent along with the orientation, and the desktop applies the turn --
    /// which also means the rotation can be corrected without rebuilding the app.
    private var orientationCode: UInt8 = 255
    private var lastStamp: CFTimeInterval = 0
    private var smoothedFPS: Double = 0

    /// Landmark order matching MediaPipe's topology, so the desktop side needs
    /// no remapping and both cameras speak the same skeleton.
    private static let landmarkOrder: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip,
    ]

    func configure(sender: PacketSender) {
        self.sender = sender
        DispatchQueue.main.async {
            UIDevice.current.beginGeneratingDeviceOrientationNotifications()
            self.updateOrientation()
            NotificationCenter.default.addObserver(
                forName: UIDevice.orientationDidChangeNotification,
                object: nil, queue: .main) { [weak self] _ in self?.updateOrientation() }
        }
    }

    private func updateOrientation() {
        switch UIDevice.current.orientation {
        case .portrait: orientationCode = 0
        case .landscapeLeft: orientationCode = 1
        case .landscapeRight: orientationCode = 2
        case .portraitUpsideDown: orientationCode = 3
        default: break                    // face up/down: keep the last known
        }
    }

    func start() {
        // Permission has to be resolved before anything touches the camera.
        // Accessing it unauthorised is not a catchable error -- the OS kills the
        // process outright, which is indistinguishable from a crash on Start.
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            queue.async { [weak self] in self?.setupAndRun() }
        case .notDetermined:
            report("requesting camera permission")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                if granted {
                    self.queue.async { self.setupAndRun() }
                } else {
                    self.report("camera permission denied")
                }
            }
        case .denied, .restricted:
            report("camera denied - enable it in Settings > HandBridge")
        @unknown default:
            report("camera permission unavailable")
        }
    }

    func stop() {
        queue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async { self?.status = "stopped" }
        }
    }

    private func setupAndRun() {
        guard session.inputs.isEmpty else {
            session.startRunning()
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .vga640x480   // matches the depth map; see below

        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera,
                                                   for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            report("no TrueDepth camera available")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else {
            report("cannot add video output"); session.commitConfiguration(); return
        }
        session.addOutput(videoOutput)

        guard session.canAddOutput(depthOutput) else {
            report("cannot add depth output"); session.commitConfiguration(); return
        }
        session.addOutput(depthOutput)
        // Filtering interpolates across holes in the dot pattern. Worth having:
        // raw TrueDepth is full of gaps at the edges of a hand, which is exactly
        // where the fingertips are.
        depthOutput.isFilteringEnabled = true
        depthOutput.alwaysDiscardsLateDepthData = true
        // Without this the output exists but delivers nothing, and the symptom is
        // a working app reporting zero depth coverage forever.
        depthOutput.connection(with: .depthData)?.isEnabled = true

        // Ask the camera to report its own intrinsics with every frame.
        if let conn = videoOutput.connection(with: .video),
           conn.isCameraIntrinsicMatrixDeliverySupported {
            conn.isCameraIntrinsicMatrixDeliveryEnabled = true
        }

        // Drive the sensor as fast as it will go. TrueDepth tops out at 30 Hz on
        // most devices; the desktop side predicts forward to cover the gap.
        //
        // Every line here is guarded because assigning activeFormat without
        // holding the device lock, or unlocking a device that was never locked,
        // both terminate the process rather than raising something catchable.
        if let format = bestFormat(for: device) {
            // activeFormat only applies when the session defers format choice to
            // the device; leaving a concrete preset set makes the assignment
            // fight with the session.
            session.sessionPreset = .inputPriority
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                device.activeFormat = format

                // Selecting the video format does NOT select a depth format,
                // and the choice matters twice over.
                //
                // First for correctness: the depth map must cover the same
                // field of view as the colour image, or the per-landmark depth
                // lookup reads from the wrong place. Aspect ratio is the
                // check available here, so it comes first.
                //
                // Then for rate: ranking by resolution picks a stream running
                // at half the video rate, so every other frame has no depth
                // partner.
                let videoAspect = Double(CMVideoFormatDescriptionGetDimensions(
                    format.formatDescription).width)
                    / Double(max(1, CMVideoFormatDescriptionGetDimensions(
                        format.formatDescription).height))

                func depthScore(_ f: AVCaptureDevice.Format) -> (Int, Double, Int) {
                    let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                    let a = d.height == 0 ? 0 : Double(d.width) / Double(d.height)
                    let matched = abs(a - videoAspect) < 0.02 ? 1 : 0
                    let rate = f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
                    return (matched, rate, Int(d.width))
                }
                let depthFormat = format.supportedDepthDataFormats
                    .max { depthScore($0) < depthScore($1) }
                if let depthFormat {
                    device.activeDepthDataFormat = depthFormat
                }

                let videoRate = format.videoSupportedFrameRateRanges
                    .map(\.maxFrameRate).max() ?? 30
                let depthRate = depthFormat?.videoSupportedFrameRateRanges
                    .map(\.maxFrameRate).max() ?? videoRate
                let scale = CMTimeScale(max(1, Int32(videoRate.rounded())))
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: scale)

                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let dsize = depthFormat
                    .map { CMVideoFormatDescriptionGetDimensions($0.formatDescription) }
                    .map { "\($0.width)x\($0.height)" } ?? "none"
                var summary = "\(dims.width)x\(dims.height)@\(Int(videoRate))  "
                    + "depth \(dsize)@\(Int(depthRate))"
                if let dd = depthFormat.map({ CMVideoFormatDescriptionGetDimensions($0.formatDescription) }) {
                    let va = Double(dims.width) / Double(max(1, dims.height))
                    let da = Double(dd.width) / Double(max(1, dd.height))
                    if abs(va - da) > 0.02 {
                        // Not cosmetic: unequal fields of view make every depth
                        // sample belong to the wrong pixel.
                        summary += "  ASPECT MISMATCH"
                    }
                }
                DispatchQueue.main.async { self.formatSummary = summary }
            } catch {
                // Not fatal: the session preset already provides a usable format,
                // just possibly a slower one.
                session.sessionPreset = .vga640x480
                report("default format (\(error.localizedDescription))")
            }
        }

        session.commitConfiguration()

        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer?.setDelegate(self, queue: queue)

        session.startRunning()
        report("streaming")
    }

    /// Prefer a depth-capable format whose ASPECT RATIO matches its depth stream.
    ///
    /// This ordering is not a preference, it is a correctness requirement.
    /// Depth is looked up per landmark by scaling colour-image pixel coordinates
    /// into depth-map coordinates. That scaling is only valid when both cover
    /// the same field of view. Pair a 16:9 colour stream with a 4:3 depth map
    /// and every landmark still receives a depth value -- so nothing looks
    /// broken -- but each one is read from the wrong part of the scene.
    ///
    /// The resulting 3D hand is distorted rather than absent, which then breaks
    /// pinch detection, screen calibration and pointing simultaneously, each
    /// looking like an independent bug.
    ///
    /// After aspect match, frame rate outranks resolution: the sensor rate sets
    /// a floor on latency that no downstream work can recover, while extra
    /// pixels only reduce landmark noise.
    private func bestFormat(for device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        func aspect(_ d: CMVideoDimensions) -> Double {
            d.height == 0 ? 0 : Double(d.width) / Double(d.height)
        }
        func score(_ f: AVCaptureDevice.Format) -> (Int, Double, Int) {
            let vd = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let va = aspect(vd)
            // Does ANY of this format's depth streams share its aspect ratio?
            let matched = f.supportedDepthDataFormats.contains { df in
                abs(aspect(CMVideoFormatDescriptionGetDimensions(df.formatDescription)) - va) < 0.02
            }
            let rate = f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            return (matched ? 1 : 0, rate, Int(vd.width))
        }
        return device.formats
            .filter { !$0.supportedDepthDataFormats.isEmpty }
            .max { a, b in score(a) < score(b) }
    }

    private func report(_ text: String) {
        DispatchQueue.main.async { self.status = text }
    }
}

extension CaptureEngine: AVCaptureDataOutputSynchronizerDelegate {

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {

        guard let videoData = collection.synchronizedData(for: videoOutput)
                as? AVCaptureSynchronizedSampleBufferData,
              !videoData.sampleBufferWasDropped,
              let pixelBuffer = CMSampleBufferGetImageBuffer(videoData.sampleBuffer)
        else { return }

        // Timestamp as early as possible: everything after this is our latency.
        let captureTime = CACurrentMediaTime()

        let depthData = (collection.synchronizedData(for: depthOutput)
                         as? AVCaptureSynchronizedDepthData)
            .flatMap { $0.depthDataWasDropped ? nil : $0.depthData }

        let intrinsics = cameraIntrinsics(from: videoData.sampleBuffer,
                                          buffer: pixelBuffer)

        // `.up` keeps Vision's normalised coordinates aligned with the raw
        // buffer, so landmark pixels, the depth map and the intrinsics all share
        // one frame of reference. Rotating here would silently desynchronise
        // them from the intrinsics, which are always reported for the raw buffer.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up, options: [:])
        try? handler.perform([handRequest])

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        guard let observation = handRequest.results?.first else {
            DispatchQueue.main.async { self.handDetected = false }
            sendPacket(landmarks: nil, depth: nil, chirality: 0,
                       width: width, height: height,
                       intrinsics: intrinsics, captureTime: captureTime)
            return
        }

        var pixels = [SIMD2<Float>](repeating: .zero, count: 21)
        var confidence = [Float](repeating: 0, count: 21)
        for (i, joint) in Self.landmarkOrder.enumerated() {
            guard let point = try? observation.recognizedPoint(joint), point.confidence > 0 else {
                continue
            }
            // Vision's origin is bottom-left; image buffers are top-left.
            pixels[i] = SIMD2<Float>(Float(point.location.x) * Float(width),
                                     Float(1.0 - point.location.y) * Float(height))
            confidence[i] = point.confidence
        }

        let depths = depthData.map { sampleDepth($0, at: pixels,
                                                 colorIntrinsics: intrinsics,
                                                 imageWidth: width, imageHeight: height) }

        let covered = depths?.filter { $0.isFinite }.count ?? 0
        DispatchQueue.main.async {
            self.handDetected = true
            self.depthCoverage = covered
        }

        sendPacket(landmarks: (pixels, confidence), depth: depths,
                   chirality: observation.chirality == .left ? 1 : 2,
                   width: width, height: height,
                   intrinsics: intrinsics, captureTime: captureTime)
    }

    /// Median depth over a small patch around each landmark, in metres.
    ///
    /// A median rather than a mean, and a patch rather than a single pixel,
    /// because a patch at the silhouette of a hand straddles the hand and
    /// whatever is metres behind it. Averaging those produces a confident value
    /// corresponding to nothing; the median stays on whichever surface fills
    /// most of the patch.
    ///
    /// Landmarks are mapped into the depth map through BOTH cameras' intrinsics
    /// rather than by scaling image coordinates. Scaling assumes the colour and
    /// depth streams share a field of view, and they generally do not -- the
    /// colour stream here is 16:9 while the depth stream is 4:3. Under that
    /// assumption every landmark still receives a depth, so nothing looks
    /// broken, but each is read from the wrong place. The visible symptom was a
    /// hand whose measured palm swung between 9 cm and 18 cm depending on
    /// whether depth had arrived, which then corrupted the learned hand model
    /// and every mode downstream of it.
    ///
    /// Un-projecting through the colour intrinsics and re-projecting through
    /// the depth intrinsics is correct for any pair of fields of view.
    private func sampleDepth(_ depthData: AVDepthData, at pixels: [SIMD2<Float>],
                             colorIntrinsics: SIMD4<Float>,
                             imageWidth: Int, imageHeight: Int) -> [Float] {
        let converted = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData
            : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = converted.depthDataMap

        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }

        let dw = CVPixelBufferGetWidth(map)
        let dh = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        guard let base = CVPixelBufferGetBaseAddress(map) else {
            return [Float](repeating: .nan, count: pixels.count)
        }

        // Depth-camera intrinsics, rescaled from the dimensions Apple reports
        // them against to the actual size of this depth map.
        var dfx: Float = 0, dfy: Float = 0, dcx: Float = 0, dcy: Float = 0
        var haveDepthIntrinsics = false
        if let calib = converted.cameraCalibrationData {
            let ref = calib.intrinsicMatrixReferenceDimensions
            let k = calib.intrinsicMatrix
            let rx = Float(dw) / Float(ref.width)
            let ry = Float(dh) / Float(ref.height)
            dfx = k.columns.0.x * rx
            dfy = k.columns.1.y * ry
            dcx = k.columns.2.x * rx
            dcy = k.columns.2.y * ry
            haveDepthIntrinsics = dfx > 1 && dfy > 1
        }
        DispatchQueue.main.async { self.depthAligned = haveDepthIntrinsics }

        // Fallback only when the depth stream reports no calibration at all.
        let sx = Float(dw) / Float(imageWidth)
        let sy = Float(dh) / Float(imageHeight)
        let radius = 2
        var out = [Float](repeating: .nan, count: pixels.count)

        for i in 0..<pixels.count {
            var mx = pixels[i].x * sx
            var my = pixels[i].y * sy
            if haveDepthIntrinsics {
                // Colour pixel -> normalised ray -> depth pixel.
                let nx = (pixels[i].x - colorIntrinsics.z) / colorIntrinsics.x
                let ny = (pixels[i].y - colorIntrinsics.w) / colorIntrinsics.y
                mx = nx * dfx + dcx
                my = ny * dfy + dcy
            }
            let cx = Int(mx.rounded())
            let cy = Int(my.rounded())
            var samples: [Float] = []
            samples.reserveCapacity((2 * radius + 1) * (2 * radius + 1))
            for dy in -radius...radius {
                let y = cy + dy
                if y < 0 || y >= dh { continue }
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
                for dx in -radius...radius {
                    let x = cx + dx
                    if x < 0 || x >= dw { continue }
                    let v = row[x]
                    // TrueDepth is specified from roughly 15 cm to 1 m; values
                    // outside that are holes or noise, not measurements.
                    if v.isFinite && v > 0.10 && v < 1.60 { samples.append(v) }
                }
            }
            if samples.count >= 3 {
                samples.sort()
                out[i] = samples[samples.count / 2]
            }
        }
        return out
    }

    private func cameraIntrinsics(from sample: CMSampleBuffer,
                                  buffer: CVPixelBuffer) -> SIMD4<Float> {
        if let attachment = CMGetAttachment(sample,
                key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
                attachmentModeOut: nil) as? Data,
           attachment.count >= MemoryLayout<matrix_float3x3>.size {
            // Copied rather than loaded in place. `load(as:)` requires the source
            // to be correctly aligned for the type, and Data makes no such
            // promise -- a misaligned load traps instead of failing, and this
            // runs on every single frame.
            var matrix = matrix_float3x3()
            _ = withUnsafeMutableBytes(of: &matrix) { dst in
                attachment.copyBytes(to: dst.bindMemory(to: UInt8.self),
                                     count: MemoryLayout<matrix_float3x3>.size)
            }
            let fx = matrix.columns.0.x, fy = matrix.columns.1.y
            let cx = matrix.columns.2.x, cy = matrix.columns.2.y
            if fx.isFinite && fy.isFinite && fx > 1 && fy > 1 {
                return SIMD4<Float>(fx, fy, cx, cy)
            }
        }
        // Fallback: a plausible focal length for a phone front camera. The
        // desktop side is told these are estimated rather than measured.
        let w = Float(CVPixelBufferGetWidth(buffer))
        let h = Float(CVPixelBufferGetHeight(buffer))
        let f = w / (2 * tan(75.0 * .pi / 180 / 2))
        return SIMD4<Float>(f, f, w / 2, h / 2)
    }

    private func sendPacket(landmarks: ([SIMD2<Float>], [Float])?,
                            depth: [Float]?, chirality: UInt8,
                            width: Int, height: Int,
                            intrinsics: SIMD4<Float>, captureTime: CFTimeInterval) {
        sequence &+= 1
        let now = CACurrentMediaTime()
        if lastStamp > 0 {
            let dt = now - lastStamp
            if dt > 0 {
                smoothedFPS = smoothedFPS == 0 ? 1 / dt : smoothedFPS * 0.9 + (1 / dt) * 0.1
                DispatchQueue.main.async { self.framesPerSecond = self.smoothedFPS }
            }
        }
        lastStamp = now

        let packet = HandPacket(orientation: orientationCode,
                                sequence: sequence,
                                captureTime: captureTime,
                                sendTime: now,
                                width: UInt16(width), height: UInt16(height),
                                intrinsics: intrinsics,
                                chirality: chirality,
                                landmarks: landmarks?.0,
                                confidence: landmarks?.1,
                                depth: depth)
        sender?.send(packet)
    }
}
