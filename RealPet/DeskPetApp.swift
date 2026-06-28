import AppKit
import Combine
import SwiftUI

@main
struct RealPetMain {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }
}

class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    var statusItem: NSStatusItem!
    var vm: PetListViewModel!
    var daemon: PythonDaemon!

    private var cancellables = Set<AnyCancellable>()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // v0.2.0: first launch may need Python+venv setup. Run the wizard
        // before starting the pipeline; otherwise jump straight to the UI.
        SetupWizard.runIfNeeded { [weak self] outcome in
            guard let self else { return }
            switch outcome {
            case .ready:
                self.startServicesAndUI()
            case .aborted(let message):
                self.showSetupFailure(message)
            }
        }
    }

    @MainActor
    private func startServicesAndUI() {
        vm = PetListViewModel()

        // Start resident Python daemon (Phase 1: detector for QC + detect).
        daemon = PythonDaemon()
        vm.pythonBridge.daemon = daemon
        daemon.onCrash = { [weak self] in
            // Auto-restart on crash (best-effort; next call will use
            // subprocess fallback if restart fails).
            self?.daemon.start()
        }
        daemon.start()

        // 创建主窗口
        let contentView = NSHostingController(
            rootView: MainPanelView(bridge: vm.pythonBridge)
                .environmentObject(vm)
        )

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 320, height: 460),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        // Manually-created NSWindows default to isReleasedWhenClosed = true,
        // which DEALLOCATES the window when the user clicks the red close
        // button — after that the menu-bar icon can no longer bring it back.
        // Keep the object alive so closing just hides it and the menu-bar icon
        // reopens it.
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.contentViewController = contentView
        window.title = "RealPet"
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.level = .floating  // 置顶
        NSApp.activate(ignoringOtherApps: true)

        // 菜单栏图标：关掉窗口后从这里重新打开
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "pawprint.fill",
                                   accessibilityDescription: "RealPet")
            button.toolTip = "RealPet — 点击显示/隐藏窗口"
            button.action = #selector(toggleWindow)
            button.target = self
        }

        // Observe pendingClipSelection to resize main window (320 ↔ 640).
        vm.$pendingClipSelection
            .receive(on: DispatchQueue.main)
            .sink { [weak self] sel in
                guard let self = self else { return }
                let w: CGFloat = sel != nil ? 640 : 320
                let h = self.window.frame.height
                self.window.setContentSize(NSSize(width: w, height: h))
                self.window.center()  // re-center after resize
            }
            .store(in: &cancellables)
    }

    @MainActor
    private func showSetupFailure(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "RealPet Setup Required"
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    // MARK: - Window delegate

    // Main window: hide instead of destroy.
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        window.orderOut(nil)
        return false
    }

    // Double-clicking the app in Finder / clicking its Dock icon when it's
    // already running (window hidden) fires this. Without it, re-launching a
    // running instance does nothing visible — which reads as "the app won't
    // open". Bring the main window back to the front.
    func applicationShouldHandleReopen(_ sender: NSApplication,
                                       hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        return true
    }

    @objc func toggleWindow() {
        if window.isVisible {
            window.orderOut(nil)
        } else {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        daemon?.terminate()
        vm?.petLauncher.stopAll()
        return .terminateNow
    }
}
