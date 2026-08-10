import SwiftUI

enum BrandColor {
    static let background = Color(hex: 0xFCF9F8)
    static let surface = Color(hex: 0xFCF9F8)
    static let surfaceDim = Color(hex: 0xDCD9D9)
    static let surfaceContainer = Color(hex: 0xF0EDED)
    static let surfaceContainerHigh = Color(hex: 0xEAE7E7)
    static let surfaceContainerLow = Color(hex: 0xF6F3F2)
    static let card = Color(hex: 0xFFF9FB)
    static let primary = Color(hex: 0x4A0E2E)
    static let primaryDeep = Color(hex: 0x2B0018)
    static let secondary = Color(hex: 0xE2B4BD)
    static let secondaryText = Color(hex: 0x79545C)
    static let textPrimary = Color(hex: 0x1C1B1B)
    static let textSecondary = Color(hex: 0x524348)
    static let outline = Color(hex: 0x847378)
    static let outlineVariant = Color(hex: 0xD6C1C7)
    static let success = Color(hex: 0x6E8B78)
}

private extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xff) / 255,
            green: Double((hex >> 8) & 0xff) / 255,
            blue: Double(hex & 0xff) / 255,
            opacity: opacity
        )
    }
}
