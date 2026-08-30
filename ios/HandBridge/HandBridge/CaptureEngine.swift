import AVFoundation
import CoreVideo
import Foundation
import QuartzCore
import simd
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
    }

    func start() {
        queue.async { [weak self] in self?.setupAndRun() }
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
        // Filtering interpolates across holes in the dot pattern. Worth having:
        // raw TrueDepth is full of gaps at the edges of a hand, which is exactly
        // where the fingertips are.
        depthOutput.isFilteringEnabled = true
        depthOutput.alwaysDiscardsLateDepthData = true
        session.addOutput(depthOutput)

        // Ask the camera to report its own intrinsics with every frame.
        if let conn = videoOutput.connection(with: .video),
           conn.isCameraIntrinsicMatrixDeliverySupported {
            conn.isCameraIntrinsicMatrixDeliveryEnabled = true
        }

        // Drive the sensor as fast as it will go. TrueDepth tops out at 30 Hz on
        // most devices; the desktop side predicts forward to cover the gap.
        if let format = bestFormat(for: device) {
            try? device.lockForConfiguration()
            device.activeFormat = format
            let maxRate = format.videoSupportedFrameRateRanges
                .map(\.maxFrameRate).max() ?? 30
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(maxRate))
            device.unlockForConfiguration()
        }

        session.commitConfiguration()

        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer?.setDelegate(self, queue: queue)

        session.startRunning()
        report("streaming")
    }

    /// Prefer a format that supports depth and runs fast. Resolution beyond the
    /// depth map's own is wasted for our purposes -- the landmark precision that
    /// matters is limited by the detector, not by pixel count.
    private func bestFormat(for device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        device.formats
            .filter { !$0.supportedDepthDataFormats.isEmpty }
            .max { a, b in
                let ra = a.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
                let rb = b.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
                if ra != rb { return ra < rb }
                let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
                let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
                return da.width < db.width
            }
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
    private func sampleDepth(_ depthData: AVDepthData, at pixels: [SIMD2<Float>],
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

        let sx = Float(dw) / Float(imageWidth)
        let sy = Float(dh) / Float(imageHeight)
        let radius = 2
        var out = [Float](repeating: .nan, count: pixels.count)

        for i in 0..<pixels.count {
            let cx = Int((pixels[i].x * sx).rounded())
            let cy = Int((pixels[i].y * sy).rounded())
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
                attachmentModeOut: nil) as? Data {
            let matrix: matrix_float3x3 = attachment.withUnsafeBytes { $0.load(as: matrix_float3x3.self) }
            return SIMD4<Float>(matrix.columns.0.x, matrix.columns.1.y,
                                matrix.columns.2.x, matrix.columns.2.y)
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

        let packet = HandPacket(sequence: sequence,
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
