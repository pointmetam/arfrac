import SwiftUI

struct ProfileView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: BrandSpacing.md) {
                profileHeader
                statCard
                draftCard
            }
            .padding(.horizontal, BrandSpacing.screenPadding)
            .padding(.vertical, BrandSpacing.md)
        }
        .background(BrandColor.background.ignoresSafeArea())
        .navigationTitle("我的")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var profileHeader: some View {
        HStack(spacing: 16) {
            Circle()
                .fill(
                    LinearGradient(
                        colors: [BrandColor.secondary, BrandColor.primary],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 72, height: 72)

            VStack(alignment: .leading, spacing: 6) {
                Text("@LeeMuse")
                    .font(.system(size: 24, weight: .bold, design: .serif))
                    .foregroundStyle(BrandColor.primaryDeep)
                Text("记录你的试妆成果与灵感收藏")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(BrandColor.textSecondary)
            }
        }
    }

    private var statCard: some View {
        HStack {
            statColumn(value: "12", title: "已发布")
            Spacer()
            statColumn(value: "28", title: "收藏")
            Spacer()
            statColumn(value: "5", title: "草稿")
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .fill(BrandColor.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .stroke(BrandColor.outlineVariant.opacity(0.75), lineWidth: 1)
        )
    }

    private var draftCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("开发中")
                .font(.system(size: 24, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primaryDeep)

            Text("这里后续会放你的作品流、收藏妆容和设置页面。当前先保留一个轻量占位，方便把主流程跑通。")
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(BrandColor.textPrimary)
                .lineSpacing(4)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                .fill(BrandColor.surfaceContainerLow)
        )
    }

    private func statColumn(value: String, title: String) -> some View {
        VStack(spacing: 6) {
            Text(value)
                .font(.system(size: 28, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primary)
            Text(title)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(BrandColor.textSecondary)
        }
    }
}

#Preview {
    NavigationStack {
        ProfileView()
    }
}
