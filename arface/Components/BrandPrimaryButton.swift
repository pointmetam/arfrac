import SwiftUI

struct BrandPrimaryButton: View {
    let title: String
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title.uppercased())
                .font(.system(size: 14, weight: .semibold))
                .tracking(1)
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .frame(height: 54)
                .background(
                    LinearGradient(
                        colors: [BrandColor.primary, BrandColor.primaryDeep],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    in: RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                )
        }
        .buttonStyle(.plain)
    }
}

struct BrandSecondaryButton: View {
    let title: String
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(BrandColor.primary)
                .frame(maxWidth: .infinity)
                .frame(height: 52)
                .background(
                    RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                        .fill(BrandColor.card)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                        .stroke(BrandColor.outlineVariant, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}
