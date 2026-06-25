import SwiftUI
import UniformTypeIdentifiers

struct MainPanelView: View {
    @EnvironmentObject var vm: PetListViewModel
    @ObservedObject var bridge: PythonBridge

    var body: some View {
        VStack(spacing: 10) {
            // Header
            HStack {
                Image(systemName: "pawprint.fill")
                    .foregroundColor(.accentColor)
                Text("RealPet")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
            }

            Divider()

            // Import button
            Button(action: { vm.showFilePicker = true }) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("Import Video")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(bridge.isProcessing)
            .fileImporter(
                isPresented: $vm.showFilePicker,
                allowedContentTypes: [
                    UTType.movie,
                    UTType.quickTimeMovie,
                    UTType.mpeg4Movie
                ],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    vm.importVideo(url: url)
                }
            }

            // Clip selection inline (long videos)
            if let selection = vm.pendingClipSelection {
                ClipSelectionView(
                    selection: selection,
                    onConfirm: { start, duration in vm.selectClip(start: start, duration: duration) },
                    onCancel: { vm.cancelClipSelection() }
                )
            }

            // Anchor confirmation — show the detected pet (green box) so the
            // user can verify before segmentation starts.
            if let det = vm.pendingDetection {
                VStack(spacing: 8) {
                    AnchorPreviewView(path: det.annotatedPath)
                        .frame(height: 180)
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Text("识别到 \(det.detectedName)（\(Int(det.detectedConf * 100))%），是否就用它？")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)

                    HStack(spacing: 12) {
                        Button("重选") { vm.cancelDetection() }
                            .buttonStyle(.borderless).foregroundColor(.red)
                        Spacer()
                        Button("就用它，开始制作") { vm.confirmDetection() }
                            .buttonStyle(.borderedProminent).controlSize(.small)
                    }
                }
                .padding(8)
                .background(Color.black.opacity(0.05))
                .cornerRadius(6)
            }

            // Progress section
            if bridge.isProcessing {
                VStack(spacing: 6) {
                    ProgressView(value: bridge.progress)
                        .progressViewStyle(.linear)

                    HStack {
                        Text(bridge.statusText)
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                        Spacer()
                        Button("Stop") {
                            bridge.cancelProcessing()
                        }
                        .buttonStyle(.borderless)
                        .font(.system(size: 11))
                        .foregroundColor(.red)
                    }
                }
                .padding(8)
                .background(Color.black.opacity(0.05))
                .cornerRadius(6)
            }

            // Error
            if let error = bridge.error {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundColor(.red)
                    .lineLimit(3)
            }

            // Pet list
            if !vm.pets.isEmpty {
                Divider()

                VStack(spacing: 0) {
                    ForEach(vm.pets) { pet in
                        PetRowView(
                            pet: pet,
                            onShow: { vm.showPet(pet) },
                            onHide: { vm.hidePet(pet) },
                            onDelete: { vm.deletePet(pet) }
                        )

                        if pet.id != vm.pets.last?.id {
                            Divider().padding(.leading, 8)
                        }
                    }
                }
            }

            Divider()

            // Quit
            Button("Quit") {
                vm.petLauncher.stopAll()
                NSApplication.shared.terminate(nil)
            }
            .buttonStyle(.borderless)
            .font(.system(size: 12))
            .foregroundColor(.secondary)
        }
        .padding(12)
        .frame(width: vm.pendingClipSelection != nil ? 640 : 300)
        .animation(.easeInOut(duration: 0.2), value: vm.pendingClipSelection != nil)
    }
}

/// Loads a thumbnail from a file path, reloading whenever the path changes.
struct ThumbImage: View {
    let path: String
    var corner: CGFloat = 6
    @State private var img: NSImage?

    var body: some View {
        Group {
            if let img {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
            } else {
                Rectangle().fill(Color.black.opacity(0.06))
            }
        }
        .cornerRadius(corner)
        .onAppear { load() }
        .onChange(of: path) { _, _ in load() }
    }

    private func load() {
        img = nil
        guard !path.isEmpty else { return }
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else { return }
        img = NSImage(data: data)
    }
}

/// Shows the annotated detection frame (green box over the pet). Reloads when
/// the path changes; the file is written fresh each detection.
struct AnchorPreviewView: View {
    let path: String
    @State private var img: NSImage?

    var body: some View {
        Group {
            if let img {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
            } else {
                Rectangle().fill(Color.black.opacity(0.06))
            }
        }
        .onAppear { load() }
        .onChange(of: path) { _, _ in load() }
    }

    private func load() {
        img = nil
        guard !path.isEmpty else { return }
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else { return }
        img = NSImage(data: data)
    }
}

/// Long-video clip picker: drag to choose a start, or tap a recommended segment.
struct ClipSelectionView: View {
    let selection: PetListViewModel.PendingClipSelection
    let onConfirm: (Double, Double) -> Void
    let onCancel: () -> Void

    private var videoAspect: Double { selection.aspect }

    // Processing-budget bounds for the selected range (seconds).
    private var minLen: Double { min(8.0, selection.videoDuration) }
    private var maxLen: Double { selection.maxSeconds }

    @State private var start: Double = 0
    @State private var end: Double = 8
    @State private var didInit = false

    private var procDuration: Double { end - start }

    private func thumbAt(_ t: Double) -> String {
        selection.filmstrip.min(by: { abs($0.t - t) < abs($1.t - t) })?.thumb ?? ""
    }

    private func fmt(_ t: Double) -> String {
        let s = Int(t.rounded()); return String(format: "%d:%02d", s / 60, s % 60)
    }

    /// Move the start handle, keeping the range within [minLen, maxLen] and
    /// inside the video. Pushes `end` along when start would cross a bound.
    private func setStart(_ v: Double) {
        let s = max(0, min(v, selection.videoDuration - minLen))
        start = s
        if end < s + minLen { end = min(selection.videoDuration, s + minLen) }
        if end > s + maxLen { end = s + maxLen }
    }

    private func setEnd(_ v: Double) {
        let e = min(selection.videoDuration, max(v, minLen))
        end = e
        if start > e - minLen { start = max(0, e - minLen) }
        if start < e - maxLen { start = e - maxLen }
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("选择片段").font(.system(size: 13, weight: .semibold))
                Spacer()
                Text("视频 \(fmt(selection.videoDuration))")
                    .font(.system(size: 11)).foregroundColor(.secondary)
            }

            // First + last frame side by side, updating live as handles move.
            HStack(spacing: 6) {
                thumbFrame(path: thumbAt(start), label: "起 \(fmt(start))", accent: true)
                thumbFrame(path: thumbAt(end), label: "止 \(fmt(end))", accent: false)
            }

            HStack {
                Text("已选时长 \(Int(procDuration.rounded())) 秒")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.accentColor)
                Spacer()
                Text("可选 \(Int(minLen))–\(Int(maxLen)) 秒")
                    .font(.system(size: 10)).foregroundColor(.secondary)
            }

            RangeSlider(start: Binding(get: { start }, set: setStart),
                        end: Binding(get: { end }, set: setEnd),
                        bounds: 0...max(minLen, selection.videoDuration))

            if !selection.candidates.isEmpty {
                Text("推荐片段")
                    .font(.system(size: 11)).foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 6) {
                    ForEach(selection.candidates) { c in
                        Button {
                            setStart(c.start)
                            setEnd(c.start + min(maxLen, max(minLen, c.duration)))
                        } label: {
                            VStack(spacing: 2) {
                                ThumbImage(path: c.thumb, corner: 4)
                                    .aspectRatio(CGFloat(videoAspect), contentMode: .fit)
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 44)
                                    .clipped()
                                Text(c.label).font(.system(size: 9))
                                Text(c.recommended ? "推荐" : " ")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.accentColor)
                            }
                            .padding(3)
                            .background(abs(start - c.start) < 0.01
                                        ? Color.accentColor.opacity(0.15) : Color.clear)
                            .cornerRadius(4)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            HStack(spacing: 12) {
                Button("取消") { onCancel() }
                    .buttonStyle(.borderless).foregroundColor(.red)
                Spacer()
                Button("用这段 \(fmt(start))–\(fmt(end))") { onConfirm(start, procDuration) }
                    .buttonStyle(.borderedProminent).controlSize(.small)
            }
        }
        .padding(8)
        .background(Color.black.opacity(0.05))
        .cornerRadius(6)
        .onAppear {
            guard !didInit else { return }
            didInit = true
            let s = selection.candidates.first(where: { $0.recommended })?.start
                ?? selection.candidates.first?.start ?? 0
            let rec = selection.candidates.first(where: { $0.recommended })
            setStart(s)
            setEnd(s + min(maxLen, max(minLen, rec?.duration ?? maxLen)))
        }
    }

    /// One start/end thumbnail with overlaid time label.
    ///
    /// Fixed display height (the image scales to fit, preserving aspect). A
    /// ratio-driven `aspectRatio(.fit)` container collapses to ~0 height inside
    /// an HStack with no height constraint, which is why the thumbnails looked
    /// tiny before. A concrete height is the reliable fix: landscape clips fill
    /// the width at this height; portrait clips render narrower but just as tall.
    @ViewBuilder
    private func thumbFrame(path: String, label: String, accent: Bool) -> some View {
        ThumbImage(path: path)
            .frame(maxWidth: .infinity)
            .frame(height: thumbHeight)
            .background(Color.black.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6)
                .stroke(accent ? Color.accentColor : Color.accentColor.opacity(0.6),
                        lineWidth: accent ? 2 : 1.5))
            .overlay(alignment: .topLeading) {
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(Color.black.opacity(0.55))
                    .foregroundColor(.white)
                    .cornerRadius(5)
                    .padding(4)
            }
    }

    /// Display height for the start/end frames. Portrait clips (aspect < 1) get
    /// a taller box so the (narrower) image is still large; landscape clips use
    /// a height that roughly fills the half-width slot at 16:9.
    private var thumbHeight: CGFloat {
        videoAspect < 1.0 ? 240 : 170
    }
}

/// A two-handle range slider (SwiftUI has no built-in one). Reports edits
/// through the bindings; the parent clamps to the processing budget.
struct RangeSlider: View {
    @Binding var start: Double
    @Binding var end: Double
    let bounds: ClosedRange<Double>

    private let handle: CGFloat = 18

    private func x(_ v: Double, _ w: CGFloat) -> CGFloat {
        let span = bounds.upperBound - bounds.lowerBound
        guard span > 0 else { return 0 }
        let frac = (v - bounds.lowerBound) / span
        return CGFloat(frac) * (w - handle) + handle / 2
    }

    private func value(_ px: CGFloat, _ w: CGFloat) -> Double {
        let span = bounds.upperBound - bounds.lowerBound
        let frac = max(0, min(1, (px - handle / 2) / (w - handle)))
        return bounds.lowerBound + Double(frac) * span
    }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            ZStack(alignment: .leading) {
                Capsule().fill(Color.gray.opacity(0.25)).frame(height: 4)
                Capsule().fill(Color.accentColor)
                    .frame(width: max(0, x(end, w) - x(start, w)), height: 4)
                    .offset(x: x(start, w))
                handleView
                    .offset(x: x(start, w) - handle / 2)
                    .gesture(DragGesture().onChanged { g in
                        start = value(g.location.x, w)
                    })
                handleView
                    .offset(x: x(end, w) - handle / 2)
                    .gesture(DragGesture().onChanged { g in
                        end = value(g.location.x, w)
                    })
            }
            .frame(height: handle)
        }
        .frame(height: handle)
    }

    private var handleView: some View {
        Circle().fill(Color.white)
            .frame(width: handle, height: handle)
            .overlay(Circle().stroke(Color.accentColor, lineWidth: 2))
            .shadow(radius: 1)
    }
}
