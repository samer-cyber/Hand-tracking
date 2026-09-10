import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import QuartzCore
import simd
import UIKit
import Vision

/// Everything the camera is asked to do, in one value so it can be replaced
/// atomically from either the phone's own UI or a command off the wire.
struct CameraSettings: Equatable {
    var fps: Double = 120
    /// Shutter as a denominator: 500 means 1/500 s.
    var shutterDenominator: Double = 500
    var isoAuto: Bool = true
    var manualISO: Float = 800
    /// Ceiling for the auto-ISO loop. Past a few thousand the sensor is mostly
    /// noise, and noise costs landmark accuracy faster than darkness does.
    var isoCeiling: Float = 4000
    var zoom: Float = 1.0
    var focusAuto: Bool = true
    var lensPosition: Float = 0.55
    /// -1 follow the device, otherwise 0 portrait / 1 landscapeLeft /
    /// 2 landscapeRight / 3 upsideDown.
    var orientationOverride: Int = -1
}

/// Rear camera to 21 landmarks, on the phone.
///
/// The design goal is that the sensor is the slowest thing in the path.
///
/// * **Rate.** A high-frame-rate rear format (1080p120 on this hardware) instead
///   of the 30 Hz default. The sampling interval is a latency floor nothing
///   downstream can recover: at 30 Hz you are on average 16 ms out of date about
///   where the hand is before any processing has even started.
/// * **Shutter.** Frame rate bounds how *often* you learn the hand's position;
///   shutter speed bounds how *smeared* each of those measurements is. A hand
///   crossing the frame in 300 ms moves several of its own widths during a
///   1/60 s exposure, and Vision then fits landmarks to a streak. 1/500 s costs
///   light, which is why there is an auto-ISO loop below.
/// * **Stabilisation off.** Video stabilisation works by holding frames back so
///   it can see where the camera went next. That is latency by construction.
/// * **Vision inline on the capture queue.** Blocking here makes AVFoundation
///   discard the next frame, which is what we want: a dropped frame costs 8 ms
///   of freshness, whereas a queued one would hand the desktop a hand position
///   that is already stale and get progressively worse.
final class CaptureEngine: NSObject, ObservableObject {

    @Published var status = "idle"
    @Published var captureFPS: Double = 0
    @Published var trackedFPS: Double = 0
    @Published var inferenceMs: Double = 0
    @Published var handDetected = false
    @Published var formatSummary = "-"
    @Published var isoValue: Float = 0
    @Published var exposureOffset: Float = 0
    @Published var orientationName = "-"

    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "handpilot.capture", qos: .userInteractive)
    private var device: AVCaptureDevice?
    private var exposureTimer: DispatchSourceTimer?

    /// Only ever touched on `queue`.
    private var settings = CameraSettings()
    private var deviceOrientation: UIDeviceOrientation = .portrait
    private var server: ControlServer?
    private var sequence: UInt32 = 0
    private var smoothedCapture: Double = 0
    private var smoothedTracked: Double = 0
    private var smoothedInference: Double = 0
    private var lastCaptureStamp: CFTimeInterval = 0
    private var lastTrackedStamp: CFTimeInterval = 0
    private var lastPublish: CFTimeInterval = 0
    private var lastStatusSend: CFTimeInterval = 0
    private var lastHandSeen: CFTimeInterval = 0

    private let request: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        // Each extra hand is a full additional pose evaluation, and against an
        // 8.3 ms frame budget that is the difference between keeping up and not.
        r.maximumHandCount = 1
        return r
    }()

    /// MediaPipe's joint ordering, so the desktop speaks one skeleton regardless
    /// of where the landmarks came from.
    private static let joints: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip,
    ]

    // MARK: - Lifecycle

    func attach(server: ControlServer) {
        self.server = server
        DispatchQueue.main.async {
            UIDevice.current.beginGeneratingDeviceOrientationNotifications()
            self.noteOrientation(UIDevice.current.orientation)
            NotificationCenter.default.addObserver(
                forName: UIDevice.orientationDidChangeNotification,
                object: nil, queue: .main) { [weak self] _ in
                    self?.noteOrientation(UIDevice.current.orientation)
                }
        }
    }

    func apply(_ new: CameraSettings) {
        queue.async { [weak self] in
            guard let self else { return }
            let formatChanged = new.fps != self.settings.fps
            self.settings = new
            if self.session.isRunning {
                if formatChanged { self.applyFormat() } else { self.applyDeviceTuning() }
                self.applyExposure(iso: nil)
            }
        }
    }

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            queue.async { [weak self] in self?.setupAndRun() }
        case .notDetermined:
            report("requesting camera permission")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                if granted { self.queue.async { self.setupAndRun() } }
                else { self.report("camera permission denied") }
            }
        default:
            // Touching the camera unauthorised is not a catchable error -- the OS
            // kills the process, which is indistinguishable from a crash on Start.
            report("camera denied - enable it in Settings > HandPilot")
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.exposureTimer?.cancel()
            self.exposureTimer = nil
            self.session.stopRunning()
            self.report("stopped")
        }
    }

    // MARK: - Session setup

    private func setupAndRun() {
        if !session.inputs.isEmpty {
            if !session.isRunning { session.startRunning() }
            startExposureLoop()
            report("streaming")
            return
        }

        session.beginConfiguration()
        // A concrete preset fights `activeFormat`; input priority hands format
        // choice to the device, which is where the 120 Hz formats live.
        session.sessionPreset = .inputPriority
        session.automaticallyConfiguresApplicationAudioSession = false

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video, position: .back)
        // The explicit wide-angle device rather than a virtual multi-camera one:
        // virtual devices switch lenses on their own, and a lens switch mid-session
        // is a discontinuity in the very coordinates the pointer is riding on.
        guard let dev = discovery.devices.first,
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration()
            report("no rear camera available")
            return
        }
        session.addInput(input)
        device = dev

        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            report("cannot add video output")
            return
        }
        session.addOutput(videoOutput)

        // YUV rather than BGRA. At 1080p120 that is roughly 370 MB/s of memory
        // traffic instead of a gigabyte, and Vision consumes either.
        let preferred = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        if videoOutput.availableVideoPixelFormatTypes.contains(preferred) {
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: preferred]
        }
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)

        if let conn = videoOutput.connection(with: .video),
           conn.isVideoStabilizationSupported {
            conn.preferredVideoStabilizationMode = .off
        }

        session.commitConfiguration()

        applyFormat()
        session.startRunning()
        startExposureLoop()
        report("streaming")
    }

    /// Pick the fastest usable rear format, preferring 1080p and 8-bit.
    ///
    /// Rate outranks resolution deliberately. Extra pixels only lower landmark
    /// noise, and Vision downsamples heavily before inference anyway; extra
    /// frames lower the latency floor, which nothing downstream can undo.
    ///
    /// 8-bit outranks 10-bit HDR for the same reason plus a second one: the HDR
    /// formats carry a tone-mapping stage the tracker gains nothing from.
    private func pickFormat(_ dev: AVCaptureDevice, targetFPS: Double) -> AVCaptureDevice.Format? {
        func rank(_ f: AVCaptureDevice.Format) -> (Int, Int, Int, Int, Int) {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let maxRate = f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let fastEnough = maxRate >= targetFPS - 0.5 ? 1 : 0
            let sub = CMFormatDescriptionGetMediaSubType(f.formatDescription)
            let eightBit = (sub == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                            || sub == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) ? 1 : 0
            let is1080 = d.height == 1080 ? 1 : 0
            // Among formats that all clear the bar, take the *slowest* one that
            // does: the 240 Hz formats bin and crop harder for a rate we have
            // explicitly asked not to use.
            return (fastEnough, eightBit, is1080, -Int(maxRate.rounded()), Int(d.width))
        }
        return dev.formats.max { rank($0) < rank($1) }
    }

    private func applyFormat() {
        guard let dev = device, let fmt = pickFormat(dev, targetFPS: settings.fps) else {
            report("no suitable camera format")
            return
        }
        do {
            try dev.lockForConfiguration()
        } catch {
            report("camera busy: \(error.localizedDescription)")
            return
        }
        dev.activeFormat = fmt

        let ranges = fmt.videoSupportedFrameRateRanges
        let maxRate = ranges.map(\.maxFrameRate).max() ?? 30
        let minRate = ranges.map(\.minFrameRate).min() ?? 1
        let fps = min(max(settings.fps, minRate), maxRate)
        // Min and max both pinned: leaving max unset lets the camera quietly drop
        // to half rate whenever it decides the scene is dark, which looks exactly
        // like the tracker slowing down for no reason.
        let dur = CMTime(value: 1, timescale: CMTimeScale(max(1, fps.rounded())))
        dev.activeVideoMinFrameDuration = dur
        dev.activeVideoMaxFrameDuration = dur

        if fmt.isVideoHDRSupported {
            dev.automaticallyAdjustsVideoHDREnabled = false
            dev.isVideoHDREnabled = false
        }
        dev.unlockForConfiguration()

        applyDeviceTuning()
        applyExposure(iso: nil)

        let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
        let summary = "\(d.width)x\(d.height) @ \(Int(fps))"
        DispatchQueue.main.async { self.formatSummary = summary }
    }

    /// Zoom and focus, which can change without reselecting a format.
    private func applyDeviceTuning() {
        guard let dev = device else { return }
        guard (try? dev.lockForConfiguration()) != nil else { return }
        defer { dev.unlockForConfiguration() }

        let ceiling = Float(min(dev.activeFormat.videoMaxZoomFactor, 4.0))
        dev.videoZoomFactor = CGFloat(min(max(settings.zoom, 1.0), max(1.0, ceiling)))

        // Refocus hunting shows up as a burst of blurred frames, and a blurred
        // frame is a frame where the fingertips land somewhere they never were.
        dev.isSubjectAreaChangeMonitoringEnabled = false
        if settings.focusAuto {
            if dev.isAutoFocusRangeRestrictionSupported { dev.autoFocusRangeRestriction = .near }
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        } else if dev.isFocusModeSupported(.locked) {
            dev.setFocusModeLockedWithLensPosition(min(max(settings.lensPosition, 0), 1),
                                                   completionHandler: nil)
        }
    }

    /// Manual shutter, and either manual or servo-driven ISO.
    private func applyExposure(iso overrideISO: Float?) {
        guard let dev = device, dev.isExposureModeSupported(.custom) else { return }
        guard (try? dev.lockForConfiguration()) != nil else { return }
        defer { dev.unlockForConfiguration() }

        let fmt = dev.activeFormat
        var dur = CMTime(value: 1,
                         timescale: CMTimeScale(max(1, settings.shutterDenominator.rounded())))
        if CMTimeCompare(dur, fmt.minExposureDuration) < 0 { dur = fmt.minExposureDuration }
        if CMTimeCompare(dur, fmt.maxExposureDuration) > 0 { dur = fmt.maxExposureDuration }

        let wanted = overrideISO ?? (settings.isoAuto ? dev.iso : settings.manualISO)
        let hi = min(fmt.maxISO, max(fmt.minISO, settings.isoCeiling))
        let iso = min(max(wanted, fmt.minISO), hi)

        dev.setExposureModeCustom(duration: dur, iso: iso, completionHandler: nil)
    }

    /// A slow servo that keeps brightness sane while the shutter stays pinned.
    ///
    /// `setExposureModeCustom` locks ISO along with shutter, so without this the
    /// picture is frozen at whatever the room happened to be when Start was
    /// pressed. The camera keeps metering regardless, and `exposureTargetOffset`
    /// reports how many stops off the current settings are -- so ISO can be
    /// servoed against it while the shutter stays exactly where the user put it.
    private func startExposureLoop() {
        exposureTimer?.cancel()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.4, repeating: 0.08)
        t.setEventHandler { [weak self] in self?.tickExposure() }
        t.resume()
        exposureTimer = t
    }

    private func tickExposure() {
        guard let dev = device, session.isRunning else { return }
        let offset = dev.exposureTargetOffset      // stops; positive = too bright
        let iso = dev.iso

        if settings.isoAuto {
            // Half-gain rather than a full correction: the meter and the servo are
            // in a loop with each other, and a full step visibly oscillates.
            let corrected = Float(Double(iso) * pow(2.0, Double(-offset) * 0.5))
            let fmt = dev.activeFormat
            let hi = min(fmt.maxISO, max(fmt.minISO, settings.isoCeiling))
            let next = min(max(corrected, fmt.minISO), hi)
            if abs(next - iso) / max(iso, 1) > 0.03 {
                applyExposure(iso: next)
            }
        }

        let now = CACurrentMediaTime()
        if now - lastPublish > 0.12 {
            lastPublish = now
            let cap = smoothedCapture, trk = smoothedTracked, inf = smoothedInference
            let hand = now - lastHandSeen < 0.25
            let orient = orientationLabel(resolvedOrientation())
            DispatchQueue.main.async {
                self.isoValue = iso
                self.exposureOffset = offset
                self.captureFPS = cap
                self.trackedFPS = trk
                self.inferenceMs = inf
                self.handDetected = hand
                self.orientationName = orient
            }
        }
        if now - lastStatusSend > 0.25 {
            lastStatusSend = now
            sendStatus(iso: iso, offset: offset)
        }
    }

    private func sendStatus(iso: Float, offset: Float) {
        guard let dev = device else { return }
        let fmt = dev.activeFormat
        let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
        let shutter = CMTimeGetSeconds(dev.exposureDuration)
        let payload: [String: Any] = [
            "capture_fps": (smoothedCapture * 10).rounded() / 10,
            "tracked_fps": (smoothedTracked * 10).rounded() / 10,
            "inference_ms": (smoothedInference * 100).rounded() / 100,
            "iso": Int(iso.rounded()),
            "iso_min": Int(fmt.minISO.rounded()),
            "iso_max": Int(fmt.maxISO.rounded()),
            "exposure_offset": (Double(offset) * 100).rounded() / 100,
            "shutter": shutter > 0 ? Int((1.0 / shutter).rounded()) : 0,
            "zoom": (Double(dev.videoZoomFactor) * 100).rounded() / 100,
            "lens": (Double(dev.lensPosition) * 100).rounded() / 100,
            "width": Int(d.width),
            "height": Int(d.height),
            "orientation": orientationCode(resolvedOrientation()),
            "orientation_name": orientationLabel(resolvedOrientation()),
            "dropped": server?.droppedCount ?? 0,
            "thermal": ProcessInfo.processInfo.thermalState.rawValue,
        ]
        server?.send(WireEncoder.json(type: Wire.typeStatus, payload), droppable: false)
    }

    // MARK: - Orientation

    private func noteOrientation(_ o: UIDeviceOrientation) {
        // Face-up and face-down carry no usable rotation, and a phone in a stand
        // reports them constantly -- so the last real orientation is kept.
        guard o == .portrait || o == .portraitUpsideDown
                || o == .landscapeLeft || o == .landscapeRight else { return }
        queue.async { [weak self] in self?.deviceOrientation = o }
    }

    private func resolvedOrientation() -> UIDeviceOrientation {
        switch settings.orientationOverride {
        case 0: return .portrait
        case 1: return .landscapeLeft
        case 2: return .landscapeRight
        case 3: return .portraitUpsideDown
        default: return deviceOrientation
        }
    }

    private func orientationCode(_ o: UIDeviceOrientation) -> Int {
        switch o {
        case .landscapeLeft: return 1
        case .landscapeRight: return 2
        case .portraitUpsideDown: return 3
        default: return 0
        }
    }

    private func orientationLabel(_ o: UIDeviceOrientation) -> String {
        switch o {
        case .landscapeLeft: return "landscape left"
        case .landscapeRight: return "landscape right"
        case .portraitUpsideDown: return "upside down"
        default: return "portrait"
        }
    }

    /// Rear-camera buffers arrive in the sensor's own landscape frame no matter
    /// how the phone is held. Telling Vision the true orientation matters for
    /// accuracy, not just bookkeeping: the hand model is trained on upright hands
    /// and degrades noticeably on a scene rotated ninety degrees.
    private func cgOrientation(_ o: UIDeviceOrientation) -> CGImagePropertyOrientation {
        switch o {
        case .portraitUpsideDown: return .left
        case .landscapeLeft: return .up
        case .landscapeRight: return .down
        default: return .right
        }
    }

    private func report(_ text: String) {
        DispatchQueue.main.async { self.status = text }
    }
}

// MARK: - Frames

extension CaptureEngine: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // The sensor's own timestamp, not "now". AVCapture stamps it on the same
        // host clock CACurrentMediaTime reads, so capture-to-send below is a real
        // measurement of this phone's contribution rather than a guess.
        let captureTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let now = CACurrentMediaTime()
        if lastCaptureStamp > 0 {
            let dt = now - lastCaptureStamp
            if dt > 0 {
                smoothedCapture = smoothedCapture == 0 ? 1 / dt
                                                       : smoothedCapture * 0.92 + (1 / dt) * 0.08
            }
        }
        lastCaptureStamp = now

        let orientation = resolvedOrientation()
        let cg = cgOrientation(orientation)

        let t0 = CACurrentMediaTime()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixel, orientation: cg, options: [:])
        // Whether it ran matters, not just what it returned: the request object is
        // reused every frame and holds onto its previous results, so swallowing a
        // failure here would quietly resend the last good hand as though it were
        // current -- a pointer that keeps drifting on stale data rather than
        // stopping, which is far harder to notice than a dropped frame.
        var performed = true
        do {
            try handler.perform([request])
        } catch {
            performed = false
        }
        let inference = (CACurrentMediaTime() - t0) * 1000
        smoothedInference = smoothedInference == 0 ? inference
                                                   : smoothedInference * 0.9 + inference * 0.1

        if lastTrackedStamp > 0 {
            let dt = now - lastTrackedStamp
            if dt > 0 {
                smoothedTracked = smoothedTracked == 0 ? 1 / dt
                                                       : smoothedTracked * 0.92 + (1 / dt) * 0.08
            }
        }
        lastTrackedStamp = now

        let bufW = CVPixelBufferGetWidth(pixel)
        let bufH = CVPixelBufferGetHeight(pixel)
        // Vision normalises against the *oriented* image, so the dimensions we
        // advertise have to be the oriented ones too -- otherwise the desktop's
        // aspect correction, which every distance threshold depends on, is
        // inverted and every pinch threshold is wrong by 16:9.
        let rotated = (cg == .right || cg == .left || cg == .rightMirrored || cg == .leftMirrored)
        let outW = UInt16(rotated ? bufH : bufW)
        let outH = UInt16(rotated ? bufW : bufH)

        var points = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: Wire.landmarkCount)
        var chirality: UInt8 = 0
        var present = false

        if performed, let observation = request.results?.first {
            present = true
            lastHandSeen = now
            switch observation.chirality {
            case .left: chirality = 1
            case .right: chirality = 2
            default: chirality = 0
            }
            // One dictionary for all 21 joints. `recognizedPoint(_:)` rebuilds
            // group lookups on every call, which is measurable at 120 Hz.
            if let all = try? observation.recognizedPoints(.all) {
                for (i, joint) in Self.joints.enumerated() {
                    guard let p = all[joint.rawValue], p.confidence > 0 else { continue }
                    // Vision's origin is bottom-left; everyone else's is top-left.
                    points[i] = SIMD3<Float>(Float(p.location.x),
                                             Float(1.0 - p.location.y),
                                             Float(p.confidence))
                }
            }
        }

        sequence &+= 1
        let frame = HandFrame(sequence: sequence,
                              captureTime: captureTime,
                              sendTime: CACurrentMediaTime(),
                              handPresent: present,
                              chirality: chirality,
                              orientation: UInt8(orientationCode(orientation)),
                              width: outW, height: outH,
                              inferenceMs: Float(inference),
                              points: points)
        server?.send(WireEncoder.frame(frame), droppable: true)
    }
}
