import SwiftUI

struct BeautyTagChip: View {
    let title: String
    var isSelected: Bool = false

    var body: some View {
        Text(title)
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(isSelected ? BrandColor.primary : BrandColor.textSecondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(
                Capsule(style: .continuous)
                    .fill(isSelected ? BrandColor.secondary.opacity(0.4) : BrandColor.card)
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(isSelected ? BrandColor.primary.opacity(0.3) : BrandColor.outlineVariant, lineWidth: 1)
            )
    }
}
