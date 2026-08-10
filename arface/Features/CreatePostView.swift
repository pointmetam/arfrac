import SwiftUI

struct CreatePostView: View {
    @State private var note = ""
    @State private var shareToCommunity = true
    @State private var syncToSocial = false

    private let tags = ["法式复古红唇", "丝缎底妆", "晚宴妆"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: BrandSpacing.md) {
                uploadCard
                captionSection
                tagSection
                toggleCard(
                    title: "分享到社区",
                    subtitle: "让更多人发现你的妆容瞬间",
                    isOn: $shareToCommunity
                )
                toggleCard(
                    title: "同步到社交平台",
                    subtitle: "微博、小红书、Instagram",
                    isOn: $syncToSocial
                )
                BrandPrimaryButton(title: "立即发布") {}
            }
            .padding(.horizontal, BrandSpacing.screenPadding)
            .padding(.vertical, BrandSpacing.md)
        }
        .background(BrandColor.background.ignoresSafeArea())
        .navigationTitle("发布成果")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var uploadCard: some View {
        VStack(spacing: 16) {
            Image(systemName: "square.and.arrow.up")
                .font(.system(size: 44, weight: .regular))
                .foregroundStyle(BrandColor.textSecondary)
            Text("点击或拖拽上传照片 / 视频")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(BrandColor.textPrimary)
            Text("支持 JPG、PNG、MP4 | 最大 50MB")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(BrandColor.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 320)
        .background(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(BrandColor.surfaceContainerLow)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(style: StrokeStyle(lineWidth: 2, dash: [8]))
                .foregroundStyle(BrandColor.outlineVariant)
        )
    }

    private var captionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("分享你的妆容心得")
                .font(.system(size: 24, weight: .bold, design: .serif))
                .foregroundStyle(BrandColor.primaryDeep)

            TextField("分享你的化妆技巧、使用的产品或者今天的心情...", text: $note, axis: .vertical)
                .lineLimit(5...8)
                .padding(18)
                .background(
                    RoundedRectangle(cornerRadius: BrandRadius.lg, style: .continuous)
                        .fill(BrandColor.card)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: BrandRadius.lg, style: .continuous)
                        .stroke(BrandColor.outlineVariant.opacity(0.8), lineWidth: 1)
                )
        }
    }

    private var tagSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("智能识别妆容", systemImage: "sparkles")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(BrandColor.primaryDeep)
                Spacer()
                Text("已自动匹配")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(BrandColor.textSecondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(BrandColor.surfaceContainer, in: Capsule())
            }

            FlowLayout(spacing: 10) {
                ForEach(tags, id: \.self) { tag in
                    BeautyTagChip(title: "# \(tag)", isSelected: true)
                }
                BeautyTagChip(title: "+ 添加标签")
            }
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

    private func toggleCard(title: String, subtitle: String, isOn: Binding<Bool>) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(BrandColor.primaryDeep)
                Text(subtitle)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(BrandColor.textSecondary)
            }
            Spacer()
            Toggle("", isOn: isOn)
                .labelsHidden()
                .tint(BrandColor.primary)
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
}

private struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let containerWidth = proposal.width ?? 300
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > containerWidth, currentX > 0 {
                currentX = 0
                currentY += rowHeight + spacing
                rowHeight = 0
            }
            rowHeight = max(rowHeight, size.height)
            currentX += size.width + spacing
        }

        return CGSize(width: containerWidth, height: currentY + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX = bounds.minX
        var currentY = bounds.minY
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.maxX, currentX > bounds.minX {
                currentX = bounds.minX
                currentY += rowHeight + spacing
                rowHeight = 0
            }

            subview.place(
                at: CGPoint(x: currentX, y: currentY),
                proposal: ProposedViewSize(width: size.width, height: size.height)
            )

            currentX += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

#Preview {
    NavigationStack {
        CreatePostView()
    }
}
