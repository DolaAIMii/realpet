import SwiftUI

struct PetRowView: View {
    let pet: Pet
    let onShow: () -> Void
    let onHide: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(pet.name)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)

                HStack(spacing: 4) {
                    if isBusy {
                        ProgressView()
                            .controlSize(.small)
                            .scaleEffect(0.6)
                            .frame(width: 10, height: 10)
                    } else {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 6, height: 6)
                    }
                    Text(statusText)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            Button(action: {
                if pet.status == .showing { onHide() } else { onShow() }
            }) {
                Image(systemName: pet.status == .showing ? "eye.slash" : "eye")
            }
            .buttonStyle(.borderless)
            .disabled(isBusy || pet.status == .failed || pet.status == .detected)
            .help(pet.status == .showing ? "Hide" : "Show")

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundColor(.red.opacity(0.7))
            }
            .buttonStyle(.borderless)
            .help("Delete")
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
    }

    private var isBusy: Bool {
        pet.status == .detecting || pet.status == .processing
    }

    private var statusColor: Color {
        switch pet.status {
        case .processing: return .orange
        case .ready: return .green
        case .showing: return .blue
        case .failed: return .red
        case .detecting: return .orange
        case .detected: return .yellow
        }
    }

    private var statusText: String {
        switch pet.status {
        case .detecting: return "准备中…"
        case .detected: return "Confirm pet"
        case .processing: return "处理中…"
        case .ready: return "Ready"
        case .showing: return "Showing"
        case .failed: return "Failed"
        }
    }
}
