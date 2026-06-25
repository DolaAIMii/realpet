import Foundation

typealias PetStatus = Pet.Status

struct Pet: Codable, Identifiable {
    let id: UUID
    var name: String
    var sourcePath: String?
    var framesDir: String?
    var frameCount: Int
    var fps: Int
    var createdAt: Date
    var status: Status

    enum Status: String, Codable {
        case detecting    // Running pet detection (~1s)
        case detected     // Detection done, waiting for user confirmation
        case processing   // Full pipeline running
        case ready        // Done, ready to show
        case showing      // Currently displayed on desktop
        case failed       // Error occurred
    }
}
