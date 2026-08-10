import SwiftUI

struct HomeView: View {
    private let look = MockData.featuredLook
    private let categories = ["全部", "派对", "日常", "复古"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: BrandSpacing.md) {
                header
                searchBar
                categoryStrip
                heroCard
                trendSection
            }
            .padding(.horizontal, BrandSpacing.screenPadding)
            .padding(.vertical, BrandSpacing.md)
        }
        .background(BrandColor.background.ignoresSafeArea())
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.hidden, for: .navigationBar)
    }

    private var header: some View {
        HStack(alignment: .center) {
            Button(action: {}) {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(BrandColor.textPrimary)
            }

            Spacer()

            Text("GLAMOUR")
                .font(.system(size: 30, weight: .bold, design: .serif))
                .tracking(1.5)
                .foregroundStyle(BrandColor.primaryDeep)

            Spacer()

            Circle()
                .fill(
                    LinearGradient(
                        colors: [BrandColor.secondary, BrandColor.primary],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 36, height: 36)
                .overlay(Circle().stroke(.white.opacity(0.8), lineWidth: 1))
        }
    }

    private var searchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(BrandColor.textSecondary)
            Text("搜索妆容、博主或趋势...")
                .foregroundStyle(BrandColor.textSecondary.opacity(0.8))
            Spacer()
        }
        .padding(.horizontal, 16)
        .frame(height: 48)
        .background(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .fill(BrandColor.surfaceContainerLow)
        )
        .overlay(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .stroke(BrandColor.outlineVariant.opacity(0.8), lineWidth: 1)
        )
    }

    private var categoryStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(Array(categories.enumerated()), id: \.offset) { index, title in
                    BeautyTagChip(title: title, isSelected: index == 0)
                }
            }
            .padding(.vertical, 2)
        }
    }

    private var heroCard: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.9),
                            BrandColor.secondary.opacity(0.75),
                            BrandColor.primary.opacity(0.94)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(height: 440)
                .overlay {
                    Image(look.heroImageName)
                        .resizable()
                        .scaledToFill()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                    .overlay {
                        LinearGradient(
                            colors: [
                                Color.black.opacity(0.08),
                                Color.black.opacity(0.16),
                                Color.black.opacity(0.46)
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                    }
                }
                .overlay(alignment: .topLeading) {
                    Circle()
                        .fill(BrandColor.secondary.opacity(0.22))
                        .frame(width: 180, height: 180)
                        .blur(radius: 10)
                        .offset(x: -35, y: -25)
                }

            VStack(alignment: .leading, spacing: 14) {
                Text("EDITORIAL PICK")
                    .font(.system(size: 12, weight: .semibold))
                    .tracking(1.4)
                    .foregroundStyle(.white.opacity(0.82))

                Text(look.title)
                    .font(.system(size: 30, weight: .bold, design: .serif))
                    .foregroundStyle(.white)

                Text(look.subtitle)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.white.opacity(0.88))
                    .lineLimit(2)

                HStack(spacing: 12) {
                    NavigationLink {
                        TryOnView(look: look)
                    } label: {
                        Label("立即试妆", systemImage: "sparkles")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(BrandColor.primary)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(.white, in: Capsule())
                    }
                    .buttonStyle(.plain)

                    NavigationLink {
                        LookDetailView(look: look)
                    } label: {
                        Label("看详情", systemImage: "play.circle")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(24)
        }
    }

    private var trendSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("探索趋势")
                    .font(.system(size: 28, weight: .bold, design: .serif))
                    .foregroundStyle(BrandColor.primaryDeep)
                Spacer()
                Text("查看全部")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(BrandColor.textSecondary)
            }

            HStack(spacing: 14) {
                trendCard(title: "玻璃肌伪素颜", accent: BrandColor.secondary)
                trendCard(title: "几何眉尾眼线", accent: BrandColor.primary)
            }
        }
    }

    private func trendCard(title: String, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            RoundedRectangle(cornerRadius: BrandRadius.lg, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [accent.opacity(0.35), accent.opacity(0.9)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(height: 180)

            Text(title)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(BrandColor.textPrimary)

            Text("快速试妆")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(BrandColor.textSecondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .fill(BrandColor.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .stroke(BrandColor.outlineVariant.opacity(0.7), lineWidth: 1)
        )
    }
}

#Preview {
    NavigationStack {
        HomeView()
    }
}
