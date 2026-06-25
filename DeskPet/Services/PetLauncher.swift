import Foundation

class PetLauncher: ObservableObject {
    private var runningPets: [UUID: Process] = [:]

    func launch(pet: Pet) {
        guard let framesDir = pet.framesDir else { return }
        launch(petId: pet.id, framesDir: framesDir, fps: pet.fps)
    }

    /// Launch (or hot-swap) a pet window from an explicit frames directory.
    /// Used both for the final pet and for the early preview (real first-N
    /// frames). If a window is already running for this pet — e.g. the
    /// preview → final swap — the old process is replaced WITHOUT emitting a
    /// spurious `.petStopped` (which would otherwise flip the pet to "ready").
    func launch(petId: UUID, framesDir: String, fps: Int) {
        guard let python = PythonBridge.findPython() else { return }

        let script = PythonBridge.projectRoot.appendingPathComponent("pipeline/pet_runner.py")

        let proc = Process()
        proc.executableURL = python
        proc.arguments = [
            script.path,
            "--frames-dir", framesDir,
            "--fps", "\(fps)"
        ]
        // pet_runner runs under SYSTEM python3 (for its built-in PyObjC/AppKit),
        // NOT the SAM2 venv. Use the augmented PATH but reset PYTHONPATH to just
        // the project root — leaking the venv's python3.10 site-packages onto a
        // python3.9 interpreter crashes numpy/cv2/PIL imports (ABI mismatch) and
        // the window silently never opens.
        var env = PythonBridge.subprocessEnvironment()
        env["PYTHONPATH"] = PythonBridge.projectRoot.path
        proc.environment = env

        proc.terminationHandler = { [weak self] endedProc in
            DispatchQueue.main.async {
                guard let self = self else { return }
                // Only report "stopped" if this exact process is still the
                // current one. A preview → final swap replaces it on purpose.
                if self.runningPets[petId] === endedProc {
                    self.runningPets.removeValue(forKey: petId)
                    NotificationCenter.default.post(
                        name: .petStopped,
                        object: nil,
                        userInfo: ["petId": petId]
                    )
                }
            }
        }

        // Overlap the swap: start the new window FIRST, retire the previous one
        // only once the new one has had time to appear. Killing the preview up
        // front leaves the desktop with no pet for the ~2s the final frames take
        // to load (80-frame Retina resize + window build), which reads as the
        // pet "disappearing" right at completion.
        let previous = runningPets[petId]
        runningPets[petId] = proc
        try? proc.run()
        if let previous = previous {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                previous.terminate()
            }
        }
    }

    func stop(petId: UUID) {
        runningPets[petId]?.terminate()
        runningPets.removeValue(forKey: petId)
    }

    func isRunning(petId: UUID) -> Bool {
        runningPets[petId]?.isRunning ?? false
    }

    func stopAll() {
        for (_, proc) in runningPets {
            proc.terminate()
        }
        runningPets.removeAll()
    }
}

extension Notification.Name {
    static let petStopped = Notification.Name("petStopped")
}
