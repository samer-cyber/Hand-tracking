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
    /// The front camera instead of the rear wide one. The rear sensor is the
    /// bigger one -- cleaner at fast shutter speeds -- but with the front camera
    /// the screen faces you.
    var useFrontCamera: Bool = false
    /// Run the hand model on a crop around where the hand was last frame rather
    /// than on the whole picture. See `CaptureEngine.regionAround`.
    var cropToHand: Bool = true
    /// The live camera picture on the phone's own screen, rather than the
    /// skeleton. Costs power and heat on a phone already working hard.
    var showPreview: Bool = true

    /// The settings as the desktop reads them back, keyed the way it sends them.
    var wireDictionary: [String: Any] {
        [
            "fps": Int(fps.rounded()),
            "shutter": Int(shutterDenominator.rounded()),
            "camera": useFrontCamera ? "front" : "back",
            "crop": cropToHand,
            "iso_auto": isoAuto,
            "iso_ceiling": Int(isoCeiling.rounded()),
            "zoom": (Double(zoom) * 100).rounded() / 100,
            "focus_auto": focusAuto,
            "lens": (Double(lensPosition) * 100).rounded() / 100,
            "orientation": orientationOverride,
            "preview": showPreview,
        ]
    }
}

/// A thinned-out copy of the hand for the phone's own screen: 30 updates a
/// second rather than 120, published on the main thread, and kept apart from
/// the engine's other published values so drawing it redraws nothing else.
final class LiveHand: ObservableObject {
    struct Snapshot {
        /// Normalised 0...1, origin top-left, exactly as sent to the computer.
        var points: [CGPoint] = []
        var visible: [Bool] = []
        var aspect: CGFloat = 16.0 / 9.0
        var present = false
    }

    @Published var snapshot = Snapshot()
}

/// Rear or front camera to 21 landmarks, on the phone.
///
/// The design goal is that the sensor is the slowest thing in the path.
///
/// * **Rate.** A high-frame-rate format (1080p120 on recent rear cameras)
///   instead of the 30 Hz default. The sampling interval is a latency floor
///   nothing downstream can recover: at 30 Hz you are on average 16 ms out of
///   date about where the hand is before any processing has even started.
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
    /// The camera currently feeding the session, so the preview can follow it.
    @Published private(set) var activeDevice: AVCaptureDevice?

    let session = AVCaptureSession()
    let liveHand = LiveHand()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "handpilot.capture", qos: .userInteractive)

    // Everything below is only ever touched on `queue`, except where noted.
    private var device: AVCaptureDevice?
    private var input: AVCaptureDeviceInput?
    private var attachedFront: Bool?
    private var outputConfigured = false
    private var exposureTimer: DispatchSourceTimer?
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
    private var lastOverlay: CFTimeInterval = 0
    private var formatMaxFPS: Double = 0
    private var nativeWidth = 0
    private var nativeHeight = 0

    /// Clockwise quarter turns reported by the rotation coordinator, for the
    /// front camera. The coordinator and its observation live on the main thread.
    private var coordinatorTurns: Int?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    private static let fullFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
    /// Where to look for the hand next frame, in Vision's normalised
    /// coordinates (origin bottom-left). The whole frame until a hand is found.
    private var searchRegion = CaptureEngine.fullFrame
    /// Whether Vision reports points relative to the region of interest (the
    /// documented behaviour) or to the whole image. Settled once, on a real
    /// frame, the first time a hand is seen -- so a wrong assumption can never
    /// put the hand in the wrong place.
    private var roiRelative: Bool?
    private var cropping = false
    /// Set if cropping proves unreliable -- a cropped request fails outright,
    /// the semantics check never gets a clear answer, or cropped frames keep
    /// losing a hand the full frame finds. Cropping then stays off for this
    /// camera rather than costing frames.
    private var cropBroken = false
    private var probeAttempts = 0
    private var cropHits = 0
    private var cropMisses = 0

    private let request: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        // Each extra hand is a full additional pose evaluation, and against an
        // 8.3 ms frame budget that is the difference between keeping up and not.
        r.maximumHandCount = 1
        return r
    }()

    private let probeRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
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

    private struct VisionPoint {
        let index: Int
        let location: CGPoint
        let confidence: Float
    }

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
            let old = self.settings
            self.settings = new
            if !new.cropToHand { self.searchRegion = Self.fullFrame }
            guard self.session.isRunning else { return }
            if new.useFrontCamera != old.useFrontCamera {
                self.attachCamera(front: new.useFrontCamera)
            } else if new.fps != old.fps {
                self.applyFormat()
            } else {
                self.applyDeviceTuning()
            }
            self.applyExposure(iso: nil)
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
            self.searchRegion = Self.fullFrame
            self.report("stopped")
        }
    }

    // MARK: - Session setup

    private func setupAndRun() {
        if input == nil || attachedFront != settings.useFrontCamera {
            guard attachCamera(front: settings.useFrontCamera) else { return }
        }
        if !session.isRunning { session.startRunning() }
        startExposureLoop()
        report("streaming")
    }

    /// The plain wide-angle camera on either side, never a virtual multi-camera
    /// device: those switch lenses on their own, and a lens switch mid-session
    /// is a discontinuity in the very coordinates the pointer is riding on.
    private func findCamera(front: Bool) -> AVCaptureDevice? {
        let position: AVCaptureDevice.Position = front ? .front : .back
        if let wide = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
            return wide
        }
        return front ? AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) : nil
    }

    /// Put the requested camera on the session, replacing whatever was there.
    @discardableResult
    private func attachCamera(front: Bool) -> Bool {
        guard let dev = findCamera(front: front) else {
            report(front ? "no front camera available" : "no rear camera available")
            return false
        }
        let newInput: AVCaptureDeviceInput
        do {
            newInput = try AVCaptureDeviceInput(device: dev)
        } catch {
            report("camera unavailable: \(error.localizedDescription)")
            return false
        }
        if front { disableCenterStage() }

        session.beginConfiguration()
        // A concrete preset fights `activeFormat`; input priority hands format
        // choice to the device, which is where the high-rate formats live.
        session.sessionPreset = .inputPriority
        session.automaticallyConfiguresApplicationAudioSession = false
        if let old = input { session.removeInput(old) }
        guard session.canAddInput(newInput) else {
            if let old = input, session.canAddInput(old) { session.addInput(old) }
            session.commitConfiguration()
            report("cannot use that camera")
            return false
        }
        session.addInput(newInput)
        if !outputConfigured {
            guard session.canAddOutput(videoOutput) else {
                session.commitConfiguration()
                report("cannot add video output")
                return false
            }
            session.addOutput(videoOutput)
            // YUV rather than BGRA. At 1080p120 that is roughly 370 MB/s of
            // memory traffic instead of a gigabyte, and Vision consumes either.
            let preferred = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            if videoOutput.availableVideoPixelFormatTypes.contains(preferred) {
                videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: preferred]
            }
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: queue)
            outputConfigured = true
        }
        if let conn = videoOutput.connection(with: .video) {
            if conn.isVideoStabilizationSupported {
                conn.preferredVideoStabilizationMode = .off
            }
            // Frames reach Vision exactly as the sensor saw them, front camera
            // included. Mirroring is the desktop's decision, made in one place.
            if conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = false
            }
        }
        session.commitConfiguration()

        input = newInput
        device = dev
        attachedFront = front
        searchRegion = Self.fullFrame
        roiRelative = nil
        cropBroken = false
        probeAttempts = 0
        cropHits = 0
        cropMisses = 0
        coordinatorTurns = nil
        applyFormat()
        watchRotation(of: dev)
        DispatchQueue.main.async { self.activeDevice = dev }
        return true
    }

    /// Center Stage crops and pans to keep faces framed, which slides the whole
    /// picture under a hand that has not moved. Off, whatever Control Center says.
    private func disableCenterStage() {
        if AVCaptureDevice.centerStageControlMode != .app {
            AVCaptureDevice.centerStageControlMode = .app
        }
        if AVCaptureDevice.isCenterStageEnabled {
            AVCaptureDevice.isCenterStageEnabled = false
        }
    }

    /// Pick the fastest usable format, preferring 1080p and 8-bit.
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
            let is1080 = min(d.width, d.height) == 1080 ? 1 : 0
            // Among formats that reach the target, take the *slowest* one that
            // does: the 240 Hz formats bin and crop harder for a rate nobody
            // asked for. When none reach it -- a front camera asked for 240 --
            // take the fastest there is.
            let rate = Int(maxRate.rounded())
            return (fastEnough, eightBit, is1080, fastEnough == 1 ? -rate : rate, Int(d.width))
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
        formatMaxFPS = maxRate

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
            dev.setFocusModeLocked(lensPosition: min(max(settings.lensPosition, 0), 1),
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
        var payload: [String: Any] = [
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
            "native_width": nativeWidth,
            "native_height": nativeHeight,
            "format_max_fps": Int(formatMaxFPS.rounded()),
            "camera_active": attachedFront == true ? "front" : "back",
            "cropping": cropping,
            "crop_unavailable": cropBroken,
            "orientation": orientationCode(resolvedOrientation()),
            "orientation_name": orientationLabel(resolvedOrientation()),
            "dropped": server?.droppedCount ?? 0,
            "thermal": ProcessInfo.processInfo.thermalState.rawValue,
            "settings": settings.wireDictionary,
        ]
        if let relative = roiRelative { payload["crop_relative"] = relative }
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

    /// For the front camera, automatic orientation follows the rotation
    /// coordinator, which knows how each model's sensor is mounted -- and that
    /// changed with the iPhone 17 generation's square front sensor.
    private func watchRotation(of dev: AVCaptureDevice) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.rotationObservation = nil
            let coordinator = AVCaptureDevice.RotationCoordinator(device: dev, previewLayer: nil)
            self.rotationCoordinator = coordinator
            self.noteCaptureAngle(coordinator.videoRotationAngleForHorizonLevelCapture)
            self.rotationObservation = coordinator.observe(
                \.videoRotationAngleForHorizonLevelCapture, options: [.new]) { [weak self] observed, _ in
                    self?.noteCaptureAngle(observed.videoRotationAngleForHorizonLevelCapture)
                }
        }
    }

    private func noteCaptureAngle(_ angle: CGFloat) {
        let turns = ((Int((angle / 90).rounded()) % 4) + 4) % 4
        queue.async { [weak self] in self?.coordinatorTurns = turns }
    }

    /// Clockwise quarter turns that make a buffer upright, from a fixed table.
    /// Camera buffers arrive in the sensor's own orientation no matter how the
    /// phone is held, and telling Vision the true orientation matters for
    /// accuracy, not just bookkeeping: the hand model is trained on upright
    /// hands and degrades noticeably on a scene rotated ninety degrees. The
    /// rear camera's table is long established; the front camera turns the
    /// other way because it faces the other way, and a portrait-mounted front
    /// sensor starts a quarter turn earlier.
    static func quarterTurns(for o: UIDeviceOrientation, front: Bool, portraitSensor: Bool) -> Int {
        let k: Int
        switch o {
        case .landscapeLeft: k = 1
        case .portraitUpsideDown: k = 2
        case .landscapeRight: k = 3
        default: k = 0
        }
        if !front { return (5 - k) % 4 }
        return ((portraitSensor ? 0 : 1) + k) % 4
    }

    static func cgOrientation(turns: Int) -> CGImagePropertyOrientation {
        switch turns {
        case 1: return .right
        case 2: return .down
        case 3: return .left
        default: return .up
        }
    }

    // MARK: - Hand crop

    /// A square around the hand with a generous margin, in Vision's normalised
    /// coordinates, or nil when a crop would not help.
    ///
    /// Vision scales whatever it is given down to a fixed working size, so the
    /// fewer pixels outside the hand, the more of that size the hand gets --
    /// the same effect as optical zoom, without giving up the wide view needed
    /// to find the hand again if it leaves. Square in pixels so the crop is not
    /// distorted; twice the hand's size and more so it survives a fast move
    /// between two frames 8 ms apart.
    private static func regionAround(_ points: [VisionPoint],
                                     imageWidth: Double, imageHeight: Double) -> CGRect? {
        let confident = points.filter { $0.confidence > 0.3 }
        guard confident.count >= 5, imageWidth > 0, imageHeight > 0 else { return nil }
        var minX = Double.infinity, minY = Double.infinity
        var maxX = -Double.infinity, maxY = -Double.infinity
        for point in confident {
            let x = Double(point.location.x) * imageWidth
            let y = Double(point.location.y) * imageHeight
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
        }
        let shortSide = min(imageWidth, imageHeight)
        let side = max(max(maxX - minX, maxY - minY) * 2.2, shortSide * 0.3)
        guard side < shortSide * 0.95 else { return nil }
        // Vision fails the whole request if the region pokes outside 0...1 by
        // even a rounding error, so stay a hair inside.
        let margin = 1e-4
        let width = side / imageWidth
        let height = side / imageHeight
        let x = min(max(((minX + maxX) / 2 - side / 2) / imageWidth, margin), 1 - width - margin)
        let y = min(max(((minY + maxY) / 2 - side / 2) / imageHeight, margin), 1 - height - margin)
        return CGRect(x: x, y: y, width: width, height: height)
    }

    /// Vision documents results as relative to the region of interest. This
    /// checks it once, on a real frame, rather than trusting it: run the same
    /// image again with a crop around the hand just found, and see which reading
    /// of the answer puts the hand back where the full-frame pass saw it.
    private func probeCropSemantics(handler: VNImageRequestHandler, found points: [VisionPoint],
                                    imageWidth: Double, imageHeight: Double) {
        guard points.count >= 8,
              let region = Self.regionAround(points, imageWidth: imageWidth,
                                            imageHeight: imageHeight),
              region.width < 0.8 || region.height < 0.8 else { return }
        // Each attempt costs a second inference on this frame; if no clear
        // answer comes after a handful, stop asking and leave cropping off.
        probeAttempts += 1
        if probeAttempts > 20 {
            cropBroken = true
            return
        }
        probeRequest.regionOfInterest = region
        guard (try? handler.perform([probeRequest])) != nil,
              let observation = probeRequest.results?.first,
              let all = try? observation.recognizedPoints(.all) else { return }

        var rawX = 0.0, rawY = 0.0, mappedX = 0.0, mappedY = 0.0, fullX = 0.0, fullY = 0.0
        var count = 0.0
        for point in points {
            guard let p = all[Self.joints[point.index]], p.confidence > 0.3 else { continue }
            let px = Double(p.location.x), py = Double(p.location.y)
            rawX += px
            rawY += py
            mappedX += Double(region.minX) + px * Double(region.width)
            mappedY += Double(region.minY) + py * Double(region.height)
            fullX += Double(point.location.x)
            fullY += Double(point.location.y)
            count += 1
        }
        guard count >= 5 else { return }
        let rawError = hypot(rawX - fullX, rawY - fullY) / count
        let mappedError = hypot(mappedX - fullX, mappedY - fullY) / count
        // Only a clear answer is kept; an ambiguous one is tried again the next
        // time a hand is found on a full-frame pass.
        if mappedError * 3 < rawError {
            roiRelative = true
        } else if rawError * 3 < mappedError {
            roiRelative = false
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

        let bufW = CVPixelBufferGetWidth(pixel)
        let bufH = CVPixelBufferGetHeight(pixel)
        nativeWidth = bufW
        nativeHeight = bufH

        let front = attachedFront == true
        let orientation = resolvedOrientation()
        let turns: Int
        if front, settings.orientationOverride < 0, let fromCoordinator = coordinatorTurns {
            turns = fromCoordinator
        } else {
            turns = Self.quarterTurns(for: orientation, front: front, portraitSensor: bufH > bufW)
        }
        let cg = Self.cgOrientation(turns: turns)

        // Vision normalises against the *oriented* image, so the dimensions we
        // advertise have to be the oriented ones too -- otherwise the desktop's
        // aspect correction, which every distance threshold depends on, is
        // inverted and every pinch threshold is wrong by 16:9.
        let rotated = turns % 2 == 1
        let outW = UInt16(rotated ? bufH : bufW)
        let outH = UInt16(rotated ? bufW : bufH)

        let usingCrop = settings.cropToHand && !cropBroken && roiRelative != nil
            && searchRegion != Self.fullFrame
        let region = searchRegion
        request.regionOfInterest = usingCrop ? region : Self.fullFrame

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
            if usingCrop { cropBroken = true }
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

        var points = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: Wire.landmarkCount)
        var visionPoints: [VisionPoint] = []
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
            //
            // The dictionary is keyed by JointName itself, not by the underlying
            // VNRecognizedPointKey that JointName wraps -- the hand-pose
            // observation specialises the generic point API.
            if let all = try? observation.recognizedPoints(.all) {
                for (i, joint) in Self.joints.enumerated() {
                    guard let p = all[joint], p.confidence > 0 else { continue }
                    var location = p.location
                    if usingCrop && roiRelative == true {
                        location = CGPoint(x: region.minX + location.x * region.width,
                                           y: region.minY + location.y * region.height)
                    }
                    visionPoints.append(VisionPoint(index: i, location: location,
                                                    confidence: p.confidence))
                    // Vision's origin is bottom-left; everyone else's is top-left.
                    points[i] = SIMD3<Float>(Float(location.x),
                                             Float(1.0 - location.y),
                                             Float(p.confidence))
                }
            }
        }

        if usingCrop {
            if present { cropHits += 1 } else { cropMisses += 1 }
            // A crop that keeps losing the hand is worse than no crop at all.
            if cropMisses >= 20 && cropMisses * 2 > cropHits { cropBroken = true }
        }
        if settings.cropToHand && !cropBroken && present {
            if roiRelative == nil && !usingCrop {
                probeCropSemantics(handler: handler, found: visionPoints,
                                   imageWidth: Double(outW), imageHeight: Double(outH))
            }
            searchRegion = Self.regionAround(visionPoints, imageWidth: Double(outW),
                                             imageHeight: Double(outH)) ?? Self.fullFrame
        } else {
            // Lost it, or cropping is off: look at the whole picture next time.
            searchRegion = Self.fullFrame
        }
        cropping = usingCrop

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

        if now - lastOverlay > 1.0 / 30.0 {
            lastOverlay = now
            var snapshot = LiveHand.Snapshot()
            snapshot.present = present
            snapshot.aspect = outH > 0 ? CGFloat(outW) / CGFloat(outH) : 16.0 / 9.0
            snapshot.points = points.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            snapshot.visible = points.map { $0.z > 0.1 }
            let live = liveHand
            DispatchQueue.main.async { live.snapshot = snapshot }
        }
    }
}
