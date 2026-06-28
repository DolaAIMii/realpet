import SwiftUI
import AppKit

/// First-launch wizard that ensures a working Python venv exists.
///
/// v0.2.0: the DMG bundles all model weights and ffmpeg, but Python itself
/// cannot be shipped inside the .app (PyTorch dylib signing + AGPL risks).
/// Instead, on first launch we guide the user to install Python via Homebrew
/// (one copy-paste command, no sudo on Apple Silicon) and then automatically
/// create the venv, install pip dependencies, install SAM2, and write a marker
/// file so subsequent launches skip this step.
final class SetupWizard: NSObject, ObservableObject {
    enum Outcome {
        case ready
        case aborted(String)
    }

    enum Stage: String {
        case checking = "Checking Python…"
        case createVenv = "Creating virtual environment…"
        case pipInstall = "Installing dependencies…"
        case sam2Install = "Installing SAM2…"
        case verifying = "Verifying…"
        case done = "Ready"
        case failed = "Setup failed"
    }

    @Published var stage: Stage = .checking
    @Published var log: [String] = []
    @Published var errorMessage: String?

    private var window: NSWindow?
    private var completion: ((Outcome) -> Void)?

    // MARK: - Entry point

    @MainActor
    static func runIfNeeded(completion: @escaping (Outcome) -> Void) {
        let fm = FileManager.default
        if let marker = try? String(contentsOf: markerURL(), encoding: .utf8),
           !marker.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           fm.isExecutableFile(atPath: URL(fileURLWithPath: marker)
               .appendingPathComponent("bin/python").path) {
            completion(.ready)
            return
        }

        let wizard = SetupWizard()
        wizard.completion = completion
        wizard.presentWindow()
    }

    // MARK: - Marker / venv location

    static func appSupportURL() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("RealPet")
    }

    static func markerURL() -> URL {
        return appSupportURL().appendingPathComponent("venv_path.txt")
    }

    static func defaultVenvDir() -> URL {
        return appSupportURL().appendingPathComponent("venv")
    }

    // MARK: - Window (MainActor)

    @MainActor
    private func presentWindow() {
        let contentView = SetupWizardView(wizard: self)

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 380),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window?.title = "RealPet Setup"
        window?.contentViewController = NSHostingController(rootView: contentView)
        window?.delegate = self
        window?.center()
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window?.isReleasedWhenClosed = false

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.performSteps()
        }
    }

    private func closeWindow() {
        DispatchQueue.main.async { [weak self] in
            self?.window?.orderOut(nil)
            self?.window = nil
        }
    }

    // MARK: - Actions (MainActor)

    @MainActor
    func retry() {
        errorMessage = nil
        log.removeAll()
        stage = .checking
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.performSteps()
        }
    }

    // MARK: - Steps (background, nonisolated)

    private func performSteps() {
        setStage(.checking)

        // 1. Find Python 3.10+
        guard let python = findPython() else {
            abort(missingPythonMessage())
            return
        }
        log("Found Python at \(python.path)")

        let venvDir = Self.defaultVenvDir()
        let venvPath = venvDir.path

        // 2. Create venv
        setStage(.createVenv)
        if !runTask(label: "create venv",
                    executable: python,
                    arguments: ["-m", "venv", venvPath]) {
            return
        }

        let venvPython = venvDir.appendingPathComponent("bin/python").path
        guard FileManager.default.isExecutableFile(atPath: venvPython) else {
            abort("Virtual environment created but \(venvPython) is not executable.")
            return
        }

        // 3. Install Python dependencies
        setStage(.pipInstall)
        guard let requirementsURL = Bundle.main.url(forResource: "requirements",
                                                    withExtension: "txt") else {
            abort("Could not find requirements.txt in the app bundle.")
            return
        }
        let pip = venvDir.appendingPathComponent("bin/pip").path
        if !runTask(label: "pip install requirements",
                    executable: URL(fileURLWithPath: pip),
                    arguments: ["install", "-r", requirementsURL.path]) {
            return
        }

        // 4. Install SAM2
        setStage(.sam2Install)
        if !runTask(label: "pip install sam2",
                    executable: URL(fileURLWithPath: pip),
                    arguments: ["install", "sam2"]) {
            return
        }

        // 5. Verify imports
        setStage(.verifying)
        let verifyScript = """
        import torch, sam2
        from transformers import AutoModelForImageSegmentation
        print('imports ok')
        """
        if !runTask(label: "verify imports",
                    executable: URL(fileURLWithPath: venvPython),
                    arguments: ["-c", verifyScript]) {
            return
        }

        // 6. Persist marker
        do {
            try FileManager.default.createDirectory(at: Self.appSupportURL(),
                                                    withIntermediateDirectories: true)
            try venvPath.write(to: Self.markerURL(), atomically: true, encoding: .utf8)
        } catch {
            abort("Failed to write setup marker: \(error.localizedDescription)")
            return
        }

        setStage(.done)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self else { return }
            self.closeWindow()
            self.completion?(.ready)
        }
    }

    // MARK: - Helpers (nonisolated, safe to call from background)

    private func findPython() -> URL? {
        let candidates = [
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3.10",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3.12",
            "/usr/local/bin/python3.11",
            "/usr/local/bin/python3.10",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            if versionOK(path) { return URL(fileURLWithPath: path) }
        }
        return nil
    }

    private func versionOK(_ path: String) -> Bool {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: path)
        proc.arguments = ["--version"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do {
            try proc.run()
            proc.waitUntilExit()
            let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                              encoding: .utf8) ?? ""
            let parts = text.trimmingCharacters(in: .whitespacesAndNewlines)
                .split(separator: " ")
            guard let versionPart = parts.last else { return false }
            let components = versionPart.split(separator: ".").compactMap { Int($0) }
            guard components.count >= 2 else { return false }
            let major = components[0]
            let minor = components[1]
            return major > 3 || (major == 3 && minor >= 10)
        } catch {
            return false
        }
    }

    private func missingPythonMessage() -> String {
        let brewInstalled = FileManager.default.isExecutableFile(atPath: "/opt/homebrew/bin/brew")
            || FileManager.default.isExecutableFile(atPath: "/usr/local/bin/brew")
        if brewInstalled {
            return """
            RealPet needs Python 3.10+ to run the AI pipeline.
            Open Terminal and paste (no sudo needed on Apple Silicon):

              brew install python@3.12

            Then click “重新检测 / Retry”.
            """
        } else {
            return """
            RealPet needs Python 3.10+ to run the AI pipeline.
            Homebrew was not detected. Install it first:

              /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

            Then install Python:

              brew install python@3.12

            Then click “重新检测 / Retry”.
            """
        }
    }

    @discardableResult
    private func runTask(label: String, executable: URL, arguments: [String]) -> Bool {
        log("→ \(label)")
        let proc = Process()
        proc.executableURL = executable
        proc.arguments = arguments
        proc.environment = Self.freshEnvironment()

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        do {
            try proc.run()
            let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
            let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()

            if let out = String(data: outData, encoding: .utf8), !out.isEmpty {
                out.split(separator: "\n").forEach { log(String($0)) }
            }
            if let err = String(data: errData, encoding: .utf8), !err.isEmpty {
                err.split(separator: "\n").forEach { log(String($0)) }
            }

            if proc.terminationStatus == 0 {
                return true
            }
            abort("\(label) failed with exit code \(proc.terminationStatus).")
            return false
        } catch {
            abort("\(label) failed: \(error.localizedDescription)")
            return false
        }
    }

    private static func freshEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let toolPaths = ["/opt/homebrew/bin", "/usr/local/bin",
                         "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        let existing = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        var merged: [String] = []
        for p in toolPaths + existing where !merged.contains(p) { merged.append(p) }
        env["PATH"] = merged.joined(separator: ":")
        return env
    }

    // MARK: - UI updates (dispatch to main internally so background callers are safe)

    private func setStage(_ stage: Stage) {
        DispatchQueue.main.async { [weak self] in
            self?.stage = stage
        }
    }

    private func log(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.log.append(message)
        }
    }

    private func abort(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.errorMessage = message
            self?.stage = .failed
        }
    }
}

extension SetupWizard: NSWindowDelegate {
    func windowWillClose(_ notification: Notification) {
        guard stage != .done else { return }
        completion?(.aborted(errorMessage ?? "Setup window was closed."))
        window = nil
    }
}

struct SetupWizardView: View {
    @ObservedObject var wizard: SetupWizard

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("RealPet Setup")
                .font(.title2)
                .bold()

            Text("First launch prepares the Python environment (~2 minutes, one-time).")
                .foregroundColor(.secondary)

            HStack(spacing: 12) {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .opacity(wizard.stage == .done || wizard.stage == .failed ? 0 : 1)
                Text(wizard.stage.rawValue)
                    .font(.headline)
            }

            if let error = wizard.errorMessage {
                ScrollView {
                    Text(error)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
                .padding(8)
                .background(Color.red.opacity(0.1))
                .cornerRadius(6)

                Button("重新检测 / Retry") {
                    wizard.retry()
                }
                .keyboardShortcut(.defaultAction)
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(wizard.log.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .frame(maxHeight: 160)
        }
        .padding()
        .frame(minWidth: 480, minHeight: 320)
    }
}
