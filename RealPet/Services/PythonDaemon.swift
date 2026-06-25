import Foundation

/// Manages a resident Python daemon process that keeps the Faster R-CNN
/// detector loaded across QC / detect calls, eliminating cold-start overhead.
///
/// Phase 1: Only `qc` and `detect` go through the daemon.  `process`
/// (SAM2 + BiRefNet) still runs as a separate subprocess.
@MainActor
final class PythonDaemon: ObservableObject {
    @Published private(set) var isReady = false
    @Published private(set) var isRunning = false

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?

    /// Registered per-request callbacks, keyed by request ID.
    private var pending: [Int: ([String: Any]) -> Void] = [:]
    private var nextId: Int = 1
    private var stdoutBuffer = ""

    /// Callback invoked when the daemon dies unexpectedly.
    var onCrash: (() -> Void)?

    // MARK: - Lifecycle

    /// Start the daemon process.  No-op if already running.
    func start() {
        guard process == nil else { return }

        guard let python = PythonBridge.findTrackMattePython() else {
            PythonBridge.log("PythonDaemon: no Python found")
            return
        }

        let script = PythonBridge.projectRoot.appendingPathComponent("scripts/daemon.py")
        let proc = Process()
        proc.executableURL = python
        proc.arguments = [script.path]
        proc.environment = PythonBridge.subprocessEnvironment()
        proc.currentDirectoryURL = PythonBridge.projectRoot

        let inp = Pipe()   // stdin → daemon
        let out = Pipe()   // daemon → stdout
        let err = Pipe()   // daemon → stderr (log only)
        proc.standardInput = inp
        proc.standardOutput = out
        proc.standardError = err

        // Read stdout in background — parse NDJSON lines.
        out.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                self?.handleStdoutChunk(chunk)
            }
        }

        // Log stderr for debugging.
        err.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let s = String(data: data, encoding: .utf8), !s.isEmpty {
                PythonBridge.log("PythonDaemon stderr: \(s)")
            }
        }

        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                guard let self else { return }
                PythonBridge.log("PythonDaemon exited with code \(p.terminationStatus)")
                self.process = nil
                self.isRunning = false
                self.isReady = false
                self.stdinPipe = nil
                self.stdoutPipe = nil
                self.stderrPipe = nil
                // Fail all pending requests
                for (_, cb) in self.pending {
                    cb(["type": "error", "message": "daemon exited unexpectedly"])
                }
                self.pending.removeAll()
                self.onCrash?()
            }
        }

        do {
            try proc.run()
        } catch {
            PythonBridge.log("PythonDaemon failed to start: \(error)")
            return
        }

        process = proc
        stdinPipe = inp
        stdoutPipe = out
        stderrPipe = err
        isRunning = true
        PythonBridge.log("PythonDaemon started pid=\(proc.processIdentifier)")
    }

    /// Terminate the daemon cleanly.
    func terminate() {
        guard let proc = process else { return }
        // Close stdin → daemon's for-loop exits → clean sys.exit(0)
        stdinPipe?.fileHandleForWriting.closeFile()
        // Give it a moment, then hard-kill if stuck.
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
            if proc.isRunning {
                proc.terminate()
            }
        }
        process = nil
        isRunning = false
        isReady = false
        pending.removeAll()
        PythonBridge.log("PythonDaemon terminated")
    }

    // MARK: - Send

    /// Send a request and stream response lines to `onLine`.
    /// The `done` line (containing `"done":true`) is also delivered via `onLine`.
    /// Returns the request ID (useful for future cancel support).
    @discardableResult
    func send(cmd: String, args: [String: Any],
              onLine: @escaping ([String: Any]) -> Void) -> Int {
        // Guard against a dead/zombie daemon: writing to a closed pipe would
        // silently swallow the request and the callback would never fire,
        // freezing the UI on "detecting". Verify the process is actually alive
        // before trusting the pipe; if not, mark not-ready and fail fast so the
        // caller falls back to the subprocess path.
        guard let proc = process, proc.isRunning, isReady,
              let handle = stdinPipe?.fileHandleForWriting else {
            isReady = false
            onLine(["type": "error", "message": "daemon not running"])
            return -1
        }

        let id = nextId
        nextId += 1

        var payload = args
        payload["id"] = id
        payload["cmd"] = cmd

        let line: String
        if let data = try? JSONSerialization.data(withJSONObject: payload),
           let s = String(data: data, encoding: .utf8) {
            line = s + "\n"
        } else {
            onLine(["type": "error", "message": "failed to encode request"])
            return -1
        }

        // Writing to a broken pipe raises SIGPIPE/throws — catch it, mark the
        // daemon dead, and fail this request so the caller can retry via
        // subprocess instead of hanging.
        pending[id] = onLine
        do {
            try handle.write(contentsOf: line.data(using: .utf8)!)
        } catch {
            pending.removeValue(forKey: id)
            isReady = false
            PythonBridge.log("PythonDaemon: write failed (\(error)), marking dead")
            onLine(["type": "error", "message": "daemon write failed"])
            return -1
        }
        return id
    }

    /// Cancel a pending request by ID (best-effort — daemon may already be
    /// processing it; Phase 1 does not support mid-request cancellation).
    func cancel(id: Int) {
        pending.removeValue(forKey: id)
    }

    // MARK: - stdout parsing

    private func handleStdoutChunk(_ chunk: String) {
        stdoutBuffer += chunk
        while let range = stdoutBuffer.range(of: "\n") {
            let line = String(stdoutBuffer[..<range.lowerBound])
            stdoutBuffer = String(stdoutBuffer[range.upperBound...])
            parseLine(line)
        }
    }

    private func parseLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            PythonBridge.log("PythonDaemon: unparseable line: \(line)")
            return
        }

        // Ready signal (id=0, type=ready) — mark daemon as ready.
        if msg["type"] as? String == "ready" {
            isReady = true
            PythonBridge.log("PythonDaemon: ready")
            return
        }

        // Route by request ID.
        guard let id = msg["id"] as? Int else { return }

        if let cb = pending[id] {
            cb(msg)
            // Remove callback when the terminal "done" line arrives.
            if msg["done"] as? Bool == true {
                pending.removeValue(forKey: id)
            }
        }
    }
}
