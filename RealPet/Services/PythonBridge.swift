import Foundation
import Combine

class PythonBridge: ObservableObject {
    @Published var isProcessing = false
    @Published var progress: Double = 0
    @Published var statusText = ""
    @Published var phaseLabel = ""
    @Published var error: String?

    private var process: Process?
    private var buffer = ""

    /// Resident daemon for QC / detect.  Set by AppDelegate after
    /// creating both objects.  When nil or not ready, callers fall back to
    /// subprocess.
    weak var daemon: PythonDaemon?

    static let projectRoot: URL = {
        let bundleURL = Bundle.main.bundleURL
        let fm = FileManager.default

        // Allow explicit override via environment variable
        if let override = ProcessInfo.processInfo.environment["REALPET_PROJECT_ROOT"],
           fm.fileExists(atPath: URL(fileURLWithPath: override).appendingPathComponent("pipeline").path) {
            return URL(fileURLWithPath: override)
        }

        if let resourcePath = Bundle.main.resourcePath {
            let candidate = URL(fileURLWithPath: resourcePath)
            if fm.fileExists(atPath: candidate.appendingPathComponent("pipeline").path) {
                return candidate
            }
        }

        var url = bundleURL
        for _ in 0..<5 {
            let candidate = url.appendingPathComponent("pipeline")
            if fm.fileExists(atPath: candidate.path) {
                return url
            }
            url = url.deletingLastPathComponent()
        }

        // If we can't find the project root, try the current working directory
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        if fm.fileExists(atPath: cwd.appendingPathComponent("pipeline").path) {
            return cwd
        }

        fputs("ERROR: Could not locate project root (pipeline/ directory). "
              + "Set REALPET_PROJECT_ROOT env var or run from the project directory.\n", stderr)
        return bundleURL
    }()

    /// Resolve the venv directory.  Order: REALPET_VENV env var → project .venv → ~/.venvs/sam2
    static var venvDir: String {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment

        // 1. Explicit env var
        if let venv = env["REALPET_VENV"], fm.fileExists(atPath: venv) {
            return venv
        }

        // 2. Project-local .venv
        let localVenv = projectRoot.appendingPathComponent(".venv").path
        if fm.fileExists(atPath: localVenv) {
            return localVenv
        }

        // 3. Legacy default
        return NSHomeDirectory() + "/.venvs/sam2"
    }

    /// Environment for Python subprocesses.
    ///
    /// Critically augments PATH with common tool locations. When the app is
    /// launched from Finder / a .app bundle it inherits launchd's minimal PATH
    /// (/usr/bin:/bin:/usr/sbin:/sbin), which omits /opt/homebrew/bin — so the
    /// pipeline's bare `ffmpeg` calls die with FileNotFoundError and the import
    /// shows up as "failed". Running via `swift run` masked this because it
    /// inherited the shell's full PATH.
    ///
    /// Also sets PYTHONPATH for the project root + venv site-packages.
    static func subprocessEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment

        let toolPaths = ["/opt/homebrew/bin", "/usr/local/bin",
                         "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        let existing = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        var merged: [String] = []
        for p in toolPaths + existing where !merged.contains(p) { merged.append(p) }
        env["PATH"] = merged.joined(separator: ":")

        let venvSitePackages = venvDir + "/lib/python3.10/site-packages"
        let existingPythonPath = env["PYTHONPATH"] ?? ""
        env["PYTHONPATH"] = "\(projectRoot.path):\(venvSitePackages):\(existingPythonPath)"
        return env
    }

    /// Find a system Python 3 (for scripts that need PyObjC, not the SAM2 venv).
    static func findSystemPython() -> URL? {
        for path in ["/usr/bin/python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3"] {
            if FileManager.default.isExecutableFile(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }

    /// Find Python for the Track-then-Matte pipeline (SAM2 venv with Python 3.10+)
    static func findTrackMattePython() -> URL? {
        let venvPython = venvDir + "/bin/python"
        if FileManager.default.isExecutableFile(atPath: venvPython) {
            return URL(fileURLWithPath: venvPython)
        }
        return nil
    }

    static func checkDeps(completion: @escaping (Bool, [String], Bool) -> Void) {
        if findTrackMattePython() != nil {
            completion(true, [], true)
        } else {
            completion(false, ["python3 (with sam2 venv)"], false)
        }
    }

    func startProcessing(videoPath: String, outputDir: String, useCoreML: Bool = true) {
        guard !isProcessing else { return }

        guard let python = Self.findTrackMattePython() else {
            error = "Python not found. Run ./install.sh first."
            return
        }
        startTrackMatte(python: python, videoPath: videoPath, outputDir: outputDir)
    }

    /// Analyze a video for clip-selection candidates (long videos). Returns the
    /// parsed `clips` payload (candidates + filmstrip + needs_selection) or nil.
    func analyzeClips(videoPath: String, outputDir: String,
                      maxSeconds: Double = 15,
                      completion: @escaping ([String: Any]?) -> Void) {
        guard let python = Self.findTrackMattePython() else {
            completion(nil); return
        }
        let script = Self.projectRoot.appendingPathComponent("scripts/analyze_clips.py")
        let proc = Process()
        proc.executableURL = python
        proc.arguments = [script.path, "--video", videoPath,
                          "--output-dir", outputDir,
                          "--max-seconds", "\(maxSeconds)"]
        proc.environment = Self.subprocessEnvironment()
        proc.currentDirectoryURL = Self.projectRoot

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice

        DispatchQueue.global(qos: .userInitiated).async {
            var result: [String: Any]? = nil
            do {
                try proc.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                // Parse the last JSON line whose type == "clips".
                let text = String(data: data, encoding: .utf8) ?? ""
                for line in text.split(separator: "\n").reversed() {
                    if let d = line.data(using: .utf8),
                       let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                       j["type"] as? String == "clips" {
                        result = j; break
                    }
                }
            } catch { result = nil }
            DispatchQueue.main.async { completion(result) }
        }
    }

    /// Run quality gate (resolution / brightness / blur / pet detection).
    /// Returns `{"type":"qc", "passed": true/false, "reason":..., "message":...}` or nil on error.
    /// Routes through the resident daemon when available, falls back to subprocess.
    @MainActor
    func qualityCheck(videoPath: String, outputDir: String,
                      completion: @escaping ([String: Any]?) -> Void) {
        // Try daemon first.
        if let daemon, daemon.isReady {
            Self.log("qualityCheck via daemon video=\(videoPath)")
            var settled = false
            daemon.send(cmd: "qc",
                        args: ["video": videoPath, "output_dir": outputDir]) { [weak self] msg in
                guard !settled else { return }
                // Daemon died / errored → fall back to subprocess.
                if msg["type"] as? String == "error" {
                    settled = true
                    Self.log("qualityCheck daemon error → subprocess fallback")
                    Task { @MainActor in
                        self?.qualityCheckSubprocess(videoPath: videoPath, outputDir: outputDir,
                                                     completion: completion)
                    }
                    return
                }
                // The terminal "done" line carries the result.
                if msg["done"] as? Bool == true {
                    settled = true
                    let result = msg["result"] as? [String: Any]
                    DispatchQueue.main.async { completion(result) }
                }
            }
            return
        }

        // Fallback: subprocess (legacy path).
        qualityCheckSubprocess(videoPath: videoPath, outputDir: outputDir, completion: completion)
    }

    /// Subprocess QC path (daemon-unavailable fallback).
    private func qualityCheckSubprocess(videoPath: String, outputDir: String,
                                        completion: @escaping ([String: Any]?) -> Void) {
        Self.log("qualityCheck via subprocess (daemon unavailable)")
        guard let python = Self.findTrackMattePython() else {
            completion(nil); return
        }
        let script = Self.projectRoot.appendingPathComponent("scripts/quality_check.py")
        let proc = Process()
        proc.executableURL = python
        proc.arguments = [script.path, "--video", videoPath, "--output-dir", outputDir]
        proc.environment = Self.subprocessEnvironment()
        proc.currentDirectoryURL = Self.projectRoot

        let pipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = errPipe

        Self.log("qualityCheck START python=\(python.path) script=\(script.path) video=\(videoPath)")
        DispatchQueue.global(qos: .userInitiated).async {
            var result: [String: Any]? = nil
            do {
                try proc.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                let text = String(data: data, encoding: .utf8) ?? ""
                let err = String(data: errData, encoding: .utf8) ?? ""
                Self.log("qualityCheck DONE exit=\(proc.terminationStatus)\nSTDOUT:\n\(text)\nSTDERR:\n\(err)")
                for line in text.split(separator: "\n").reversed() {
                    if let d = line.data(using: .utf8),
                       let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                       j["type"] as? String == "qc" {
                        result = j; break
                    }
                }
            } catch {
                Self.log("qualityCheck THREW: \(error)")
                result = nil
            }
            DispatchQueue.main.async { completion(result) }
        }
    }

    /// Append a line to /tmp/realpet_pybridge.log for debugging the .app
    /// (where stdout/stderr aren't visible).
    static func log(_ msg: String) {
        let line = "[\(Date())] \(msg)\n"
        let path = "/tmp/realpet_pybridge.log"
        if let h = FileHandle(forWritingAtPath: path) {
            h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); h.closeFile()
        } else {
            try? line.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    /// Run quick pet detection and return result via callback.
    /// Routes through the resident daemon when available, falls back to subprocess.
    @MainActor
    func detectPet(videoPath: String, outputDir: String, startTime: Double = 0,
                   completion: @escaping ([String: Any]?) -> Void) {
        // Try daemon first.
        if let daemon, daemon.isReady {
            Self.log("detectPet via daemon video=\(videoPath) start=\(startTime)")
            var args: [String: Any] = ["video": videoPath, "output_dir": outputDir]
            if startTime > 0 { args["start"] = startTime }
            var settled = false
            daemon.send(cmd: "detect", args: args) { [weak self] msg in
                guard !settled else { return }
                // Daemon died / errored mid-request → fall back to subprocess
                // instead of leaving the caller hanging on "detecting".
                if msg["type"] as? String == "error" {
                    settled = true
                    Self.log("detectPet daemon error → subprocess fallback")
                    Task { @MainActor in
                        self?.detectPetSubprocess(videoPath: videoPath, outputDir: outputDir,
                                                  startTime: startTime, completion: completion)
                    }
                    return
                }
                if msg["done"] as? Bool == true {
                    settled = true
                    let result = msg["result"] as? [String: Any]
                    DispatchQueue.main.async { completion(result) }
                }
            }
            return
        }

        // Fallback: subprocess (legacy path).
        detectPetSubprocess(videoPath: videoPath, outputDir: outputDir,
                            startTime: startTime, completion: completion)
    }

    /// Subprocess detect path (daemon-unavailable fallback).
    private func detectPetSubprocess(videoPath: String, outputDir: String, startTime: Double,
                                     completion: @escaping ([String: Any]?) -> Void) {
        Self.log("detectPet via subprocess (daemon unavailable)")
        guard let python = Self.findTrackMattePython() else {
            completion(nil)
            return
        }

        let script = Self.projectRoot.appendingPathComponent("scripts/detect_pet.py")
        let proc = Process()
        proc.executableURL = python
        var detectArgs = [script.path, "--video", videoPath, "--output-dir", outputDir]
        if startTime > 0 { detectArgs += ["--start", "\(startTime)"] }
        proc.arguments = detectArgs
        proc.environment = Self.subprocessEnvironment()
        proc.currentDirectoryURL = Self.projectRoot

        let pipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = errPipe

        DispatchQueue.global(qos: .userInitiated).async {
            var parsed: [String: Any]? = nil
            do {
                try proc.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    parsed = json
                } else {
                    let out = String(data: data, encoding: .utf8) ?? ""
                    let err = String(data: errData, encoding: .utf8) ?? ""
                    let log = "[detectPet] exit=\(proc.terminationStatus)\nSTDOUT:\n\(out)\nSTDERR:\n\(err)\n"
                    try? log.write(toFile: "/tmp/realpet_pybridge.log", atomically: true, encoding: .utf8)
                }
            } catch {
                try? "[detectPet] proc.run() threw: \(error)\n".write(
                    toFile: "/tmp/realpet_pybridge.log", atomically: true, encoding: .utf8)
            }
            DispatchQueue.main.async { completion(parsed) }
        }
    }

    func startWithClick(videoPath: String, outputDir: String, clickX: Int, clickY: Int,
                        startTime: Double = -1, duration: Double = -1) {
        guard !isProcessing, let python = Self.findTrackMattePython() else { return }

        let script = Self.projectRoot.appendingPathComponent("scripts/track_then_matte.py")
        let proc = Process()
        proc.executableURL = python
        var procArgs = [
            script.path,
            "--video", videoPath,
            "--output-dir", outputDir,
            "--fps", "10",
            "--preview-seconds", "5",
            "--max-seconds", "15",
            "--click", "\(clickX),\(clickY)"
        ]
        // User picked a specific segment → process from there (skip auto-select).
        if startTime >= 0 {
            procArgs += ["--start", "\(startTime)"]
            if duration > 0 { procArgs += ["--duration", "\(duration)"] }
        }
        proc.arguments = procArgs
        proc.environment = Self.subprocessEnvironment()

        startProcess(proc)
    }

    private func startTrackMatte(python: URL, videoPath: String, outputDir: String) {
        let script = Self.projectRoot.appendingPathComponent("scripts/track_then_matte.py")
        fputs("DEBUG: track-matte python=\(python.path) script=\(script.path)\n", stderr)

        let proc = Process()
        proc.executableURL = python
        proc.arguments = [
            script.path,
            "--video", videoPath,
            "--output-dir", outputDir,
            "--fps", "10",
            "--preview-seconds", "5",
            "--max-seconds", "15"
        ]
        proc.environment = Self.subprocessEnvironment()

        startProcess(proc)
    }

    private func startProcess(_ proc: Process) {
        // Run from the project root so the scripts' relative pipeline imports
        // resolve correctly. When launched from a .app the default
        // CWD is "/" — the scripts can't find modules there,
        // and the import shows up as "failed".
        proc.currentDirectoryURL = Self.projectRoot

        let pipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = errPipe

        buffer = ""
        isProcessing = true
        progress = 0
        error = nil
        statusText = "Starting..."

        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let errStr = String(data: data, encoding: .utf8) {
                fputs("PYTHON STDERR: \(errStr)", stderr)
            }
        }

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            self?.handleOutput(chunk)
        }

        proc.terminationHandler = { [weak self] p in
            DispatchQueue.main.async {
                self?.isProcessing = false
                if p.terminationStatus != 0 && self?.error == nil {
                    self?.error = "Process exited with code \(p.terminationStatus)"
                }
            }
        }

        self.process = proc
        try? proc.run()
    }

    func cancelProcessing() {
        process?.interrupt()
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            if self?.process?.isRunning == true {
                self?.process?.terminate()
            }
        }
    }

    private func handleOutput(_ chunk: String) {
        buffer += chunk
        while let range = buffer.range(of: "\n") {
            let line = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.upperBound...])
            parseLine(line)
        }
    }

    private func parseLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = msg["type"] as? String else { return }

        DispatchQueue.main.async {
            switch type {
            case "phase":
                if let name = msg["name"] as? String {
                    self.phaseLabel = self.phaseDisplayName(name)
                }

            case "progress":
                if let phase = msg["phase"] as? String,
                   let current = msg["current"] as? Int,
                   let total = msg["total"] as? Int, total > 0 {
                    self.progress = Double(current) / Double(total)
                    self.statusText = "\(self.phaseLabel) \(current)/\(total)"

                    // Emit preview-ready when SAM2 tracking completes first N frames
                    if phase == "sam2_track" && current == total {
                        // SAM2 done — preview video should be available soon
                    }
                }

            case "quality_failed":
                if let result = msg["result"] as? [String: Any],
                   let issues = result["issues"] as? [[String: Any]] {
                    let messages = issues.compactMap { $0["message"] as? String }
                    self.error = messages.joined(separator: "\n")
                    self.statusText = "Quality check failed"
                }
                self.isProcessing = false

            case "segmentation_poor":
                let message = msg["message"] as? String ?? "分割质量不达标"
                self.error = message
                self.statusText = "分割质量不达标"
                self.isProcessing = false
                // Tell the VM to retract any preview pet already on the desktop
                // and mark the pet failed (no "complete" will follow — the
                // pipeline hard-returns after this event).
                NotificationCenter.default.post(
                    name: .segmentationPoor, object: nil, userInfo: msg)

            case "complete":
                self.progress = 1.0
                self.statusText = "Done!"
                self.isProcessing = false
                NotificationCenter.default.post(
                    name: .processingComplete,
                    object: nil,
                    userInfo: msg
                )

            case "preview_ready":
                self.statusText = "Preview ready, refining..."
                NotificationCenter.default.post(
                    name: .previewReady,
                    object: nil,
                    userInfo: msg
                )

            case "error":
                if let message = msg["message"] as? String {
                    self.error = message
                    self.statusText = "Error: \(message)"
                }
                self.isProcessing = false

            default:
                break
            }
        }
    }

    private func phaseDisplayName(_ name: String) -> String {
        switch name {
        case "extract": return "Extracting frames"
        case "quality_check": return "Checking quality"
        case "detect": return "Detecting pet"
        case "sam2_track": return "Tracking pet"
        case "preview_video": return "Generating preview"
        case "birefnet_refine": return "Refining edges"
        case "final_video": return "Generating video"
        case "segment": return "Segmenting"
        case "normalize": return "Normalizing"
        default: return name
        }
    }
}

extension Notification.Name {
    static let processingComplete = Notification.Name("processingComplete")
    static let previewReady = Notification.Name("previewReady")
    static let segmentationPoor = Notification.Name("segmentationPoor")
}
