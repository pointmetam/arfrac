import SwiftUI

struct TryOnView: View {
    let look: FeaturedLook
    @Environment(\.openURL) private var openURL
    @StateObject private var makeupState = MakeupState()
    @State private var hasConfiguredTryOn = false
    @State private var isPanelCollapsed = false
    private let previewZoomOutScale: CGFloat = 0.95

    var body: some View {
        ZStack(alignment: .bottom) {
            previewArea
                .ignoresSafeArea()

            bottomBar
        }
        .background(BrandColor.primaryDeep.ignoresSafeArea())
        .navigationTitle("试妆")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .onAppear(perform: configureForProductTryOn)
    }

    private var previewArea: some View {
        ZStack(alignment: .topLeading) {
            ARPreviewContainer(state: makeupState)
                .ignoresSafeArea()
                .scaleEffect(previewZoomOutScale)
                .clipped()

            LinearGradient(
                colors: [.black.opacity(0.28), .clear, .black.opacity(0.56)],
                startPoint: .top,
                endPoint: .bottom
            )
            .allowsHitTesting(false)

            VStack(alignment: .leading, spacing: 8) {
                Text(look.title)
                    .font(.system(size: 23, weight: .bold, design: .serif))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                Text(makeupState.status.title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(makeupState.status.tint)
                if !makeupState.photoText.isEmpty {
                    Text(makeupState.photoText)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.82))
                }
            }
            .padding(24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var bottomBar: some View {
        VStack(spacing: isPanelCollapsed ? 10 : 13) {
            panelHandle

            HStack(spacing: 8) {
                ForEach(tryOnLooks) { shade in
                    Button {
                        makeupState.selectedLook = shade
                    } label: {
                        VStack(spacing: 6) {
                            Circle()
                                .fill(shade.swatch)
                                .frame(width: 28, height: 28)
                                .overlay {
                                    Circle()
                                        .stroke(.white.opacity(makeupState.selectedLook.id == shade.id ? 1 : 0.35), lineWidth: 2)
                                        .padding(2)
                                }
                            Text(shade.shortName)
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(makeupState.selectedLook.id == shade.id ? BrandColor.primary : BrandColor.textSecondary)
                        }
                        .frame(maxWidth: .infinity, minHeight: 52)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }

            if !isPanelCollapsed {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(makeupState.selectedLook.name)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(BrandColor.primaryDeep)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                        Spacer()
                        Text("\(Int(makeupState.intensity * 100))%")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(BrandColor.textSecondary)
                    }

                    Slider(value: $makeupState.intensity, in: 0...1)
                        .tint(BrandColor.primary)
                }
            }

            HStack {
                circularButton(systemName: makeupState.isMakeupVisible ? "eye" : "eye.slash") {
                    makeupState.isMakeupVisible.toggle()
                }
                Spacer()
                Button(action: requestPhotoCapture) {
                    ZStack {
                        Circle()
                            .fill(BrandColor.card.opacity(0.35))
                            .frame(width: 78, height: 78)
                        Circle()
                            .stroke(BrandColor.card.opacity(0.9), lineWidth: 3)
                            .frame(width: 66, height: 66)
                        Circle()
                            .fill(BrandColor.primary)
                            .frame(width: 56, height: 56)
                        VStack(spacing: 2) {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(.white)
                            Text("拍照")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundStyle(.white.opacity(0.92))
                        }
                    }
                }
                .buttonStyle(.plain)
                Spacer()
                circularButton(systemName: "arrow.clockwise") {
                    resetMakeup()
                }
            }

            if !isPanelCollapsed {
                HStack(spacing: 10) {
                    BrandSecondaryButton(title: "看教程") {
                        openURL(look.tutorialURL)
                    }
                    NavigationLink {
                        CreatePostView()
                    } label: {
                        Text("发布同款")
                            .font(.system(size: 14, weight: .semibold))
                            .tracking(1)
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 44)
                            .background(
                                RoundedRectangle(cornerRadius: BrandRadius.xl, style: .continuous)
                                    .fill(BrandColor.primary)
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(.horizontal, BrandSpacing.screenPadding)
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(
            RoundedRectangle(cornerRadius: 32, style: .continuous)
                .fill(BrandColor.background.opacity(0.94))
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private var panelHandle: some View {
        VStack(spacing: 6) {
            Capsule()
                .fill(BrandColor.outlineVariant.opacity(0.95))
                .frame(width: 42, height: 5)
            Image(systemName: isPanelCollapsed ? "chevron.up" : "chevron.down")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(BrandColor.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 2)
        .contentShape(Rectangle())
        .onTapGesture {
            togglePanelCollapsed()
        }
        .gesture(
            DragGesture(minimumDistance: 12)
                .onEnded { value in
                    if value.translation.height > 18 {
                        setPanelCollapsed(true)
                    } else if value.translation.height < -18 {
                        setPanelCollapsed(false)
                    }
                }
        )
    }

    private func circularButton(systemName: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(BrandColor.primary)
                .frame(width: 54, height: 54)
                .background(.white, in: Circle())
        }
        .buttonStyle(.plain)
    }

    private func configureForProductTryOn() {
        makeupState.isMakeupVisible = true
        makeupState.showFaceMesh = false
        makeupState.isLipOnlyTestMode = true
        makeupState.forceRawUVOnly = false
        makeupState.isRecordingARKitUV = false
        makeupState.isRecordingLip2D = false
        makeupState.lipMode = .lipSDFRefinement

        // Keep the user's current shade if this view appears again while the AR session is active.
        let selectedLookStillVisible = tryOnLooks.contains { $0.id == makeupState.selectedLook.id }
        if !hasConfiguredTryOn || !selectedLookStillVisible {
            if let defaultLook = defaultTryOnLook {
                makeupState.selectedLook = defaultLook
            }
        }
        hasConfiguredTryOn = true
    }

    private func resetMakeup() {
        makeupState.isMakeupVisible = true
        makeupState.intensity = 0.88
        makeupState.isLipOnlyTestMode = true
        makeupState.forceRawUVOnly = false
        makeupState.lipMode = .lipSDFRefinement
        makeupState.photoText = ""
        if let defaultLook = defaultTryOnLook {
            makeupState.selectedLook = defaultLook
        }
    }

    private func requestPhotoCapture() {
        makeupState.photoText = "正在保存照片..."
        makeupState.photoCaptureRequestID += 1
    }

    private func togglePanelCollapsed() {
        setPanelCollapsed(!isPanelCollapsed)
    }

    private func setPanelCollapsed(_ collapsed: Bool) {
        withAnimation(.easeOut(duration: 0.18)) {
            isPanelCollapsed = collapsed
        }
    }

    private var tryOnLooks: [MakeupLook] {
        Array(makeupState.visibleLooks.prefix(4))
    }

    private var defaultTryOnLook: MakeupLook? {
        tryOnLooks.first(where: { $0.id == "lancome-242" }) ?? tryOnLooks.dropFirst().first ?? tryOnLooks.first
    }
}

#Preview {
    NavigationStack {
        TryOnView(look: MockData.featuredLook)
    }
}
