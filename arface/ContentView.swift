//
//  ContentView.swift
//  arface
//
//  Created by lee on 2026/5/21.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var makeupState = MakeupState()

    var body: some View {
        ZStack {
            ARPreviewContainer(state: makeupState)
                .ignoresSafeArea()

            LinearGradient(
                colors: [.black.opacity(0.45), .clear, .black.opacity(0.62)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 0) {
                topBar
                Spacer()
                controlPanel
            }
            .padding(.horizontal, 18)
            .padding(.top, 18)
            .padding(.bottom, 14)

            if makeupState.isTrainingRecordingActive {
                recordingGuideOverlay
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            }
        }
        .preferredColorScheme(.dark)
    }

    private var topBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("AR 全脸试妆")
                    .font(.system(size: 26, weight: .semibold))
                Text(makeupState.status.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(makeupState.status.tint)
                if (makeupState.showFaceMesh || makeupState.lipMode.usesModel), !makeupState.performanceText.isEmpty {
                    Text(makeupState.performanceText)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.72))
                }
                if !makeupState.recordingText.isEmpty && !makeupState.isTrainingRecordingActive {
                    Text(makeupState.recordingText)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.72))
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer()

            HStack(spacing: 10) {
                Button {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.82)) {
                        makeupState.isMakeupVisible.toggle()
                    }
                } label: {
                    Image(systemName: makeupState.isMakeupVisible ? "eye" : "eye.slash")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(makeupState.isMakeupVisible ? "隐藏妆效" : "显示妆效")

                Button {
                    makeupState.showFaceMesh.toggle()
                } label: {
                    Image(systemName: makeupState.showFaceMesh ? "face.smiling.inverse" : "face.smiling")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("分区调试")

                Button {
                    if makeupState.isRecordingLip2D {
                        makeupState.isRecordingLip2D = false
                    } else {
                        makeupState.isRecordingARKitUV = false
                        makeupState.lip2DCameraPosition = .back
                        makeupState.markLip2DRecordingButtonTapped()
                        makeupState.recordingText = "2D REC 正在启动后摄高清录制..."
                        makeupState.isRecordingLip2D = true
                    }
                } label: {
                    Image(systemName: makeupState.isRecordingLip2D ? "stop.circle.fill" : "video.badge.plus")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(makeupState.isTrainingRecordingActive ? .red : .primary)
                        .frame(width: 44, height: 44)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(makeupState.trainingRecordingAccessibilityLabel)

                Image(systemName: makeupState.lipMode.usesModel ? "sparkles" : "sparkle")
                    .font(.system(size: 18, weight: .semibold))
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
                    .accessibilityLabel("口红模型状态")
            }
        }
    }

    private var recordingGuideOverlay: some View {
        VStack {
            Spacer()
                .frame(height: 86)

            VStack(alignment: .leading, spacing: 10) {
                ForEach(recordingGuideLines, id: \.self) { line in
                    Text(line)
                        .font(line.contains("REC") ? .system(size: 24, weight: .heavy, design: .monospaced) : .system(size: 22, weight: .bold))
                        .foregroundStyle(line.contains("REC") ? .red : .white)
                        .lineLimit(2)
                        .minimumScaleFactor(0.72)
                }
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.black.opacity(0.64), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(.red.opacity(0.58), lineWidth: 1.5)
            )
            .padding(.horizontal, 18)

            Spacer()
        }
        .allowsHitTesting(false)
    }

    private var recordingGuideLines: [String] {
        makeupState.recordingText
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
    }

    private var controlPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(makeupState.selectedLook.name)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.primary)

            lookPicker
            lipModePicker

            VStack(alignment: .leading, spacing: 9) {
                HStack {
                    Text("妆效强度")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Text("\(Int(makeupState.intensity * 100))%")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                Slider(value: $makeupState.intensity, in: 0...1)
                    .tint(makeupState.selectedLook.accentColor)
            }

            Button {
                withAnimation(.spring(response: 0.25, dampingFraction: 0.82)) {
                    makeupState.isLipOnlyTestMode.toggle()
                }
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: makeupState.isLipOnlyTestMode ? "paintbrush.pointed.fill" : "face.smiling")
                        .font(.system(size: 15, weight: .bold))
                    Text(makeupState.isLipOnlyTestMode ? "只看口红" : "全脸妆效")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                }
                .frame(maxWidth: .infinity, minHeight: 42, alignment: .leading)
                .padding(.horizontal, 14)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(makeupState.isLipOnlyTestMode ? "当前只看口红" : "当前显示全脸妆效")

            Button {
                makeupState.photoCaptureRequestID += 1
                makeupState.photoText = "正在保存照片..."
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 15, weight: .bold))
                    Text(photoButtonTitle)
                        .font(.system(size: 14, weight: .semibold))
                        .lineLimit(2)
                        .minimumScaleFactor(0.78)
                    Spacer()
                }
                .frame(maxWidth: .infinity, minHeight: 42, alignment: .leading)
                .padding(.horizontal, 14)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .buttonStyle(.plain)
            .disabled(makeupState.isTrainingRecordingActive)
            .opacity(makeupState.isTrainingRecordingActive ? 0.48 : 1)
            .accessibilityLabel("拍照保存")

            HStack(spacing: 10) {
                ForEach(makeupState.visibleLooks) { look in
                    Button {
                        makeupState.selectedLook = look
                    } label: {
                        Circle()
                            .fill(look.swatch)
                            .frame(width: 34, height: 34)
                            .overlay {
                                if makeupState.selectedLook.id == look.id {
                                    Circle()
                                        .stroke(.white, lineWidth: 3)
                                        .padding(2)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(look.name)
                }

                Spacer()

                Button {
                    makeupState.isMakeupVisible = true
                    makeupState.intensity = 0.88
                    makeupState.selectedLook = makeupState.looks[1]
                    makeupState.lipMode = .lipSDFRefinement
                    makeupState.forceRawUVOnly = false
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.system(size: 16, weight: .bold))
                        .frame(width: 42, height: 42)
                        .background(.thinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("重置")
            }
        }
        .padding(18)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
            .stroke(.white.opacity(0.16), lineWidth: 1)
        )
    }

    private var photoButtonTitle: String {
        if makeupState.isTrainingRecordingActive {
            return "录制中不可拍照"
        }
        if makeupState.photoText.isEmpty {
            return "拍照保存"
        }
        return makeupState.photoText
    }

    private var lookPicker: some View {
        Picker("妆容", selection: $makeupState.selectedLook) {
            ForEach(makeupState.visibleLooks) { look in
                Text(look.shortName).tag(look)
            }
        }
        .pickerStyle(.segmented)
    }

    private var lipModePicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(LipMakeupMode.visibleModes) { mode in
                    let isSelected = makeupState.lipMode == mode
                    Button {
                        withAnimation(.spring(response: 0.22, dampingFraction: 0.86)) {
                            makeupState.lipMode = mode
                        }
                    } label: {
                        Text(mode.pickerTitle)
                            .font(.system(size: 13, weight: .semibold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.82)
                            .frame(minWidth: 48, minHeight: 34)
                            .padding(.horizontal, 8)
                            .foregroundStyle(isSelected ? .black : .white.opacity(0.86))
                            .background(
                                isSelected ? Color.white.opacity(0.92) : Color.white.opacity(0.14),
                                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                            )
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(mode.title)
                }
            }
        }
    }
}
