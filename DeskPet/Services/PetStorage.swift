import Foundation

class PetStorage {
    static let shared = PetStorage()

    let appSupportURL: URL
    let petsDir: URL
    let petsFile: URL

    private init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        appSupportURL = base.appendingPathComponent("RealPet")
        petsDir = appSupportURL.appendingPathComponent("pets")
        petsFile = appSupportURL.appendingPathComponent("pets.json")

        try? FileManager.default.createDirectory(at: petsDir, withIntermediateDirectories: true)
    }

    func petDirectory(for id: UUID) -> URL {
        petsDir.appendingPathComponent(id.uuidString)
    }

    func load() -> [Pet] {
        guard let data = try? Data(contentsOf: petsFile),
              let pets = try? JSONDecoder().decode([Pet].self, from: data) else {
            return []
        }
        return pets
    }

    func save(_ pets: [Pet]) {
        guard let data = try? JSONEncoder().encode(pets) else { return }
        try? data.write(to: petsFile, options: .atomic)
    }

    func deletePet(_ pet: Pet) {
        let dir = petDirectory(for: pet.id)
        try? FileManager.default.removeItem(at: dir)
    }
}
