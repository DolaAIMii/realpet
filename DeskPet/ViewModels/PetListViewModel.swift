import Foundation
import SwiftUI

@MainActor
class PetListViewModel: ObservableObject {
    @Published var pets: [Pet] = []
    @Published var showFilePicker = false

    /// Pending clip selection (long videos) — user picks where the clip starts
    @Published var pendingClipSelection: PendingClipSelection?

    /// Pending detection confirmation — shows the annotated (green-box) frame so
    /// the user can verify the right pet was found before segmentation starts.
    @Published var pendingDetection: PendingDetection?

    let pythonBridge = PythonBridge()
    let petLauncher = PetLauncher()

    /// Pets currently showing the early preview on the desktop (real first-N
    /// frames). On completion these are hot-swapped to the full final frames.
    private var previewing: Set<UUID> = []

    struct ClipChoice: Identifiable {
        let id = UUID()
        let start: Double
        let end: Double
        let duration: Double
        let score: Double
        let recommended: Bool
        let thumb: String
        let label: String
    }

    struct FilmFrame: Identifiable {
        let id = UUID()
        let t: Double
        let thumb: String
        let label: String
    }

    struct PendingClipSelection: Equatable {
        let petId: UUID
        let videoPath: String
        let outputDir: String
        let videoDuration: Double
        let maxSeconds: Double
        let aspect: Double
        let candidates: [ClipChoice]
        let filmstrip: [FilmFrame]

        static func == (lhs: PendingClipSelection, rhs: PendingClipSelection) -> Bool {
            lhs.petId == rhs.petId
        }
    }

    /// Detection result awaiting user confirmation. Carries the annotated frame
    /// (green box over the detected pet) plus the anchor point + the segment to
    /// process once confirmed.
    struct PendingDetection: Equatable {
        let petId: UUID
        let videoPath: String
        let outputDir: String
        let annotatedPath: String
        let detectedName: String
        let detectedConf: Double
        let clickX: Int
        let clickY: Int
        let startTime: Double
        let duration: Double

        static func == (lhs: PendingDetection, rhs: PendingDetection) -> Bool {
            lhs.petId == rhs.petId && lhs.annotatedPath == rhs.annotatedPath
        }
    }

    init() {
        pets = PetStorage.shared.load()
        NotificationCenter.default.addObserver(
            self, selector: #selector(onProcessingComplete(_:)),
            name: .processingComplete, object: nil
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(onPreviewReady(_:)),
            name: .previewReady, object: nil
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(onPetStopped(_:)),
            name: .petStopped, object: nil
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(onSegmentationPoor(_:)),
            name: .segmentationPoor, object: nil
        )
    }

    func importVideo(url: URL) {
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }

        let id = UUID()
        let petDir = PetStorage.shared.petDirectory(for: id)
        try? FileManager.default.createDirectory(at: petDir, withIntermediateDirectories: true)

        let destPath = petDir.appendingPathComponent(url.lastPathComponent).path
        do {
            try FileManager.default.copyItem(atPath: url.path, toPath: destPath)
        } catch {
            pythonBridge.error = "Failed to copy video: \(error.localizedDescription)"
            return
        }

        let name = url.deletingPathExtension().lastPathComponent
        let pet = Pet(
            id: id,
            name: name,
            sourcePath: destPath,
            framesDir: nil,
            frameCount: 0,
            fps: 10,
            createdAt: Date(),
            status: .detecting
        )
        pets.append(pet)
        PetStorage.shared.save(pets)

        // Step 0: Quality gate — reject bad videos before clip analysis.
        pythonBridge.qualityCheck(videoPath: destPath, outputDir: petDir.path) { [weak self] qcResult in
            guard let self = self else { return }
            if let qc = qcResult, (qc["passed"] as? Bool) == false {
                let reason = qc["message"] as? String ?? "素材不合格"
                self.updatePetStatus(id: id, status: .failed)
                self.pythonBridge.error = reason
                return
            }
            // QC passed → continue to clip analysis.
            self.pythonBridge.analyzeClips(videoPath: destPath, outputDir: petDir.path) { [weak self] clips in
            guard let self = self else { return }
            if let clips = clips, (clips["needs_selection"] as? Bool) == true {
                let cands = (clips["candidates"] as? [[String: Any]] ?? []).map { c in
                    ClipChoice(start: c["start"] as? Double ?? 0,
                               end: c["end"] as? Double ?? 0,
                               duration: c["duration"] as? Double ?? 0,
                               score: c["score"] as? Double ?? 0,
                               recommended: c["recommended"] as? Bool ?? false,
                               thumb: c["thumb"] as? String ?? "",
                               label: c["label"] as? String ?? "")
                }
                let strip = (clips["filmstrip"] as? [[String: Any]] ?? []).map { f in
                    FilmFrame(t: f["t"] as? Double ?? 0,
                              thumb: f["thumb"] as? String ?? "",
                              label: f["label"] as? String ?? "")
                }
                self.pendingClipSelection = PendingClipSelection(
                    petId: id, videoPath: destPath, outputDir: petDir.path,
                    videoDuration: clips["duration"] as? Double ?? 0,
                    maxSeconds: clips["max_seconds"] as? Double ?? 15,
                    aspect: clips["aspect"] as? Double ?? (16.0 / 9.0),
                    candidates: cands, filmstrip: strip)
            } else {
                // Short video / analysis unavailable → use the whole clip.
                self.runDetection(petId: id, videoPath: destPath,
                                  outputDir: petDir.path, startTime: -1, duration: -1)
            }
        }
        } // QC gate closure
    }

    /// User picked a clip segment → detect the anchor on that segment and
    /// auto-confirm (no manual confirmation page).
    func selectClip(start: Double, duration: Double) {
        guard let sel = pendingClipSelection else { return }
        pendingClipSelection = nil
        runDetection(petId: sel.petId, videoPath: sel.videoPath,
                     outputDir: sel.outputDir, startTime: start, duration: duration)
    }

    /// Run pet detection on the chosen segment's first frame, then surface the
    /// annotated (green-box) frame for the user to confirm before segmenting.
    private func runDetection(petId: UUID, videoPath: String, outputDir: String,
                              startTime: Double, duration: Double) {
        updatePetStatus(id: petId, status: .detecting)
        pythonBridge.detectPet(videoPath: videoPath, outputDir: outputDir,
                               startTime: max(0, startTime)) { [weak self] result in
            DispatchQueue.main.async {
                guard let self = self, let result = result else {
                    self?.updatePetStatus(id: petId, status: .failed)
                    return
                }
                let type = result["type"] as? String
                if type == "detected" {
                    let cx = result["cx"] as? Int ?? 0
                    let cy = result["cy"] as? Int ?? 0
                    // Show the green-box confirmation so the user can verify the
                    // right pet was found (and reject if wrong) before we spend
                    // minutes segmenting the wrong subject.
                    self.pendingDetection = PendingDetection(
                        petId: petId, videoPath: videoPath, outputDir: outputDir,
                        annotatedPath: result["annotated"] as? String ?? "",
                        detectedName: result["name"] as? String ?? "pet",
                        detectedConf: result["conf"] as? Double ?? 0,
                        clickX: cx, clickY: cy,
                        startTime: startTime, duration: duration)
                    self.updatePetStatus(id: petId, status: .detected)
                } else if type == "no_pet" {
                    // QC gate should have caught this, but keep as safety net.
                    self.updatePetStatus(id: petId, status: .failed)
                } else {
                    self.updatePetStatus(id: petId, status: .failed)
                }
            }
        }
    }

    /// User confirmed the detected pet → start segmentation.
    func confirmDetection() {
        guard let det = pendingDetection else { return }
        pendingDetection = nil
        updatePetStatus(id: det.petId, status: .processing)
        if det.clickX >= 0 && det.clickY >= 0 {
            pythonBridge.startWithClick(
                videoPath: det.videoPath, outputDir: det.outputDir,
                clickX: det.clickX, clickY: det.clickY,
                startTime: det.startTime, duration: det.duration)
        } else {
            pythonBridge.startProcessing(videoPath: det.videoPath, outputDir: det.outputDir)
        }
    }

    /// User rejected the detection → mark failed, drop the pending state.
    func cancelDetection() {
        guard let det = pendingDetection else { return }
        pendingDetection = nil
        updatePetStatus(id: det.petId, status: .failed)
    }

    /// User cancelled clip selection
    func cancelClipSelection() {
        guard let sel = pendingClipSelection else { return }
        updatePetStatus(id: sel.petId, status: .failed)
        pendingClipSelection = nil
    }

    /// Update pet status by id
    private func updatePetStatus(id: UUID, status: PetStatus) {
        if let idx = pets.firstIndex(where: { $0.id == id }) {
            pets[idx].status = status
            PetStorage.shared.save(pets)
        }
    }

    /// Preview ready — drop the pet onto the desktop early using the real
    /// first-N matted frames (same model as final, so it's WYSIWYG).
    @objc private func onPreviewReady(_ notification: Notification) {
        guard let info = notification.userInfo as? [String: Any],
              let previewDir = info["preview_dir"] as? String else { return }

        guard let idx = pets.lastIndex(where: { $0.status == .processing }) else { return }
        let pet = pets[idx]
        previewing.insert(pet.id)
        petLauncher.launch(petId: pet.id, framesDir: previewDir, fps: pet.fps)
    }

    /// Final processing complete
    @objc private func onProcessingComplete(_ notification: Notification) {
        guard let info = notification.userInfo as? [String: Any] else { return }

        let framesDir = info["frames_dir"] as? String
            ?? info["segmented_dir"] as? String
        let frameCount = info["frame_count"] as? Int ?? info["frames"] as? Int

        guard let framesDir = framesDir, let frameCount = frameCount else { return }

        if let idx = pets.lastIndex(where: { $0.status == .processing }) {
            pets[idx].framesDir = framesDir
            pets[idx].frameCount = frameCount

            if previewing.remove(pets[idx].id) != nil {
                // Already on the desktop as a preview — seamlessly swap to the
                // full final frames and keep it showing.
                petLauncher.launch(petId: pets[idx].id, framesDir: framesDir,
                                   fps: pets[idx].fps)
                pets[idx].status = .showing
            } else {
                pets[idx].status = .ready
            }
            PetStorage.shared.save(pets)
        }
    }

    /// Post-processing gate failed (too many red frames). The pipeline hard-
    /// returns without emitting "complete", so no pet should land. Retract any
    /// preview pet already on the desktop and mark this pet failed.
    @objc private func onSegmentationPoor(_ notification: Notification) {
        guard let idx = pets.lastIndex(where: { $0.status == .processing }) else { return }
        let pet = pets[idx]
        // Pull the early-preview pet off the desktop if it's showing.
        if previewing.remove(pet.id) != nil {
            petLauncher.stop(petId: pet.id)
        }
        pets[idx].status = .failed
        PetStorage.shared.save(pets)
    }

    @objc private func onPetStopped(_ notification: Notification) {
        guard let info = notification.userInfo as? [String: Any],
              let petId = info["petId"] as? UUID else { return }
        // User dismissed the window — don't auto-reshow it when final completes.
        previewing.remove(petId)
        // Only a finished, on-screen pet flips back to "ready". If the preview
        // was closed mid-processing, leave the status as .processing.
        if let idx = pets.firstIndex(where: { $0.id == petId }),
           pets[idx].status == .showing {
            pets[idx].status = .ready
        }
    }

    func showPet(_ pet: Pet) {
        guard let idx = pets.firstIndex(where: { $0.id == pet.id }) else { return }
        guard pets[idx].status != .showing else { return }  // already showing, no double-launch
        guard pet.framesDir != nil else { return }  // nothing to show yet
        pets[idx].status = .showing
        petLauncher.launch(pet: pet)
    }

    func hidePet(_ pet: Pet) {
        petLauncher.stop(petId: pet.id)
        if let idx = pets.firstIndex(where: { $0.id == pet.id }) {
            pets[idx].status = .ready  // always reset, even if process was already dead
        }
    }

    func deletePet(_ pet: Pet) {
        petLauncher.stop(petId: pet.id)
        PetStorage.shared.deletePet(pet)
        pets.removeAll { $0.id == pet.id }
        PetStorage.shared.save(pets)
    }

    func isPetRunning(_ pet: Pet) -> Bool {
        petLauncher.isRunning(petId: pet.id)
    }
}
