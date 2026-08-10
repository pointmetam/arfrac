import SwiftUI

struct LookDetailView: View {
    let look: FeaturedLook
    @Environment(\.openURL) private var openURL

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: BrandSpacing.md) {
                hero
                titleBlock
                tagSection
                actionSection
                specSection
                tutorialSection
                productSection
            }
            .padding(.horizontal, BrandSpacing.screenPadding)
            .padding(.bottom, BrandSpacing.lg)
        }
        .background(BrandColor.background.ignoresSafeArea())
        .navigationTitle("妆容详情")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var hero: some View {
        RoundedRectangle(cornerRadius: 28, style: .continuous)
            .fill(
                LinearGradient(
                    colors: [
                        BrandColor.surface,
                        BrandColor.secondary.opacity(0.65),
                        BrandColor.primary.opacity(0.95)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .frame(height: 420)
            .overlay {
                Image(look.heroImageName)
                    .resizable()
                    .scaledToFill()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                .overlay {
                    LinearGradient(
                        colors: [
                            Color.black.opacity(0.06),
                            Color.black.opacity(0.14),
                            Color.black.opacity(0.38)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                }
            }
            .overlay(alignment: .bottomLeading) {
                Text("EDITORIAL LOOK")
                    .font(.system(size: 12, weight: .semibold))
                    .tracking(1.6)
                    .foregroundStyle(.white.opacity(0.84))
                    .padding(20)
            }
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(look.title)
                .font(.system(size: 32, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primaryDeep)

            Text("由 \(look.creatorName) 创作")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(BrandColor.textSecondary)

            Text(look.subtitle)
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(BrandColor.textPrimary)
                .lineSpacing(4)
        }
    }

    private var tagSection: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(Array(look.tags.enumerated()), id: \.offset) { index, tag in
                    BeautyTagChip(title: tag, isSelected: index == 0)
                }
            }
        }
    }

    private var actionSection: some View {
        VStack(spacing: 12) {
            NavigationLink {
                TryOnView(look: look)
            } label: {
                Text("立即试这个妆")
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

            BrandSecondaryButton(title: "去小红书看教程") {
                openURL(look.tutorialURL)
            }
        }
    }

    private var specSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("妆效参数")
                .font(.system(size: 24, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primaryDeep)

            ForEach(look.specs) { spec in
                HStack {
                    Text(spec.title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(BrandColor.textSecondary)
                    Spacer()
                    Text(spec.value)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(BrandColor.textPrimary)
                }
                .padding(.vertical, 4)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .fill(BrandColor.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .stroke(BrandColor.outlineVariant.opacity(0.75), lineWidth: 1)
        )
    }

    private var tutorialSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("观看博主教程")
                .font(.system(size: 24, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primaryDeep)

            HStack(spacing: 16) {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [BrandColor.secondary, BrandColor.primary],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 54, height: 54)

                VStack(alignment: .leading, spacing: 6) {
                    Text("跟妆这套午夜玫瑰妆")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(BrandColor.textPrimary)
                    Text("小红书教程 · \(look.creatorName)")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(BrandColor.textSecondary)
                }

                Spacer()
            }

            BrandSecondaryButton(title: "去看教程") {
                openURL(look.tutorialURL)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .fill(BrandColor.surfaceContainerLow)
        )
    }

    private var productSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("所用好物清单")
                .font(.system(size: 24, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primaryDeep)

            ForEach(look.products) { product in
                HStack(spacing: 14) {
                    RoundedRectangle(cornerRadius: BrandRadius.md, style: .continuous)
                        .fill(BrandColor.secondary.opacity(0.35))
                        .frame(width: 48, height: 48)

                    VStack(alignment: .leading, spacing: 4) {
                        Text(product.name)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(BrandColor.textPrimary)
                        Text(product.category)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(BrandColor.textSecondary)
                    }

                    Spacer()

                    Text(product.price)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(BrandColor.primary)
                }
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: BrandRadius.lg, style: .continuous)
                        .fill(BrandColor.card)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: BrandRadius.lg, style: .continuous)
                        .stroke(BrandColor.outlineVariant.opacity(0.75), lineWidth: 1)
                )
            }
        }
    }
}

#Preview {
    NavigationStack {
        LookDetailView(look: MockData.featuredLook)
    }
}
