//
//  ARKitUVRecordingGuide.swift
//  arface
//
//  Created by Codex on 2026/5/29.
//

import Foundation

struct ARKitUVRecordingPhase: Equatable {
    let id: String
    let title: String
    let instruction: String
    let durationSeconds: Int
}

struct ARKitUVRecordingGuideState: Equatable {
    let phase: ARKitUVRecordingPhase
    let phaseIndex: Int
    let elapsedSeconds: Int
    let totalSeconds: Int
    let phaseRemainingSeconds: Int
    let totalRemainingSeconds: Int
    let isFinished: Bool
}

struct ARKitUVRecordingGuide: Equatable {
    static let `default` = ARKitUVRecordingGuide(phases: [
        ARKitUVRecordingPhase(id: "prepare", title: "准备", instruction: "正脸对准，保持清楚", durationSeconds: 3),
        ARKitUVRecordingPhase(id: "front-closed", title: "A 正脸闭嘴", instruction: "闭嘴放松，正脸停住", durationSeconds: 17),
        ARKitUVRecordingPhase(id: "open-close-speak", title: "B 张嘴闭嘴", instruction: "慢慢张嘴、闭嘴，说几句短句", durationSeconds: 20),
        ARKitUVRecordingPhase(id: "left-turn", title: "C 左转头", instruction: "慢慢转到左侧，再回正", durationSeconds: 15),
        ARKitUVRecordingPhase(id: "right-turn", title: "D 右转头", instruction: "慢慢转到右侧，再回正", durationSeconds: 15),
        ARKitUVRecordingPhase(id: "up-down", title: "E 低头抬头", instruction: "轻微低头、回正，再轻微抬头", durationSeconds: 15),
        ARKitUVRecordingPhase(id: "expression", title: "F 表情嘴角", instruction: "微笑、抿嘴、嘴角变化", durationSeconds: 20),
        ARKitUVRecordingPhase(id: "lighting-free", title: "G 光照/轻动作", instruction: "当前光照下轻微转头和张嘴", durationSeconds: 20),
        ARKitUVRecordingPhase(id: "final-hold", title: "收尾", instruction: "正脸停住，准备保存", durationSeconds: 5)
    ])

    let phases: [ARKitUVRecordingPhase]

    var totalDuration: Int {
        phases.reduce(0) { $0 + $1.durationSeconds }
    }

    func state(atElapsedSeconds rawElapsedSeconds: Int) -> ARKitUVRecordingGuideState {
        let total = totalDuration
        let elapsed = min(max(rawElapsedSeconds, 0), total)
        let finished = elapsed >= total
        var phaseStart = 0

        for (index, phase) in phases.enumerated() {
            let phaseEnd = phaseStart + phase.durationSeconds
            if elapsed < phaseEnd || index == phases.count - 1 {
                let remaining = finished ? 0 : max(phaseEnd - elapsed, 0)
                return ARKitUVRecordingGuideState(
                    phase: phase,
                    phaseIndex: index,
                    elapsedSeconds: elapsed,
                    totalSeconds: total,
                    phaseRemainingSeconds: remaining,
                    totalRemainingSeconds: max(total - elapsed, 0),
                    isFinished: finished
                )
            }
            phaseStart = phaseEnd
        }

        let fallback = phases[phases.count - 1]
        return ARKitUVRecordingGuideState(
            phase: fallback,
            phaseIndex: phases.count - 1,
            elapsedSeconds: elapsed,
            totalSeconds: total,
            phaseRemainingSeconds: 0,
            totalRemainingSeconds: 0,
            isFinished: true
        )
    }

    func statusText(elapsedSeconds: Int, sampleCount: Int) -> String {
        let current = state(atElapsedSeconds: elapsedSeconds)
        if current.isFinished {
            return "REC \(current.elapsedSeconds)/\(current.totalSeconds)s  保存中\n样本 \(sampleCount)"
        }

        return "REC \(current.elapsedSeconds)/\(current.totalSeconds)s  \(current.phase.title)\n\(current.phase.instruction)  剩余 \(current.phaseRemainingSeconds)s  样本 \(sampleCount)"
    }
}
