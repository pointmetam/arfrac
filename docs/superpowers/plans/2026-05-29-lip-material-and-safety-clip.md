# Lip Material And Safety Clip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UV lipstick look closer to Perfect Makeup: cleaner outside edge, preserved lip crease/highlight, less flat color fill.

**Architecture:** Keep the current UV student mask as the placement signal. Add a stricter ARKit-derived UV safety clip at render time so mask spill outside plausible lips is discarded, then render lipstick as layered material instead of one opaque fill.

**Tech Stack:** Swift, UIKit/CoreGraphics UV texture rendering, ARKit face geometry, existing Core ML UV lip student.

---

### Task 1: Add Masked Lip Safety Clip

**Files:**
- Modify: `/Users/lee/CodeBuddy/leyi/arface/arface/UVMakeupTexture.swift`

- [x] Add a narrower `lipPaintClip` path to `DynamicLipMask`.
- [x] Use `lipPaintClip` when `drawMaskedLips` receives an alpha mask.
- [x] Keep the existing `safetyClip` available for debug/guides and non-mask fallback.

### Task 2: Make Masked Lip Rendering Layered

**Files:**
- Modify: `/Users/lee/CodeBuddy/leyi/arface/arface/UVMakeupTexture.swift`

- [x] Replace the current full-rect masked fill with a small transparency layer.
- [x] Draw a softer base color through the model alpha mask.
- [x] Add a stronger central lip pass clipped by the same mask.
- [x] Restore a subtle crease darkening and lower-lip glossy highlight.

### Task 3: Verify On Device

**Files:**
- Verify: `/Users/lee/CodeBuddy/leyi/arface/arface/UVMakeupTexture.swift`

- [x] Build with `xcodebuild -workspace arface.xcworkspace -scheme arface -configuration Debug -destination id=00008101-001D694A2E82001E -allowProvisioningUpdates build`.
- [x] Install with `xcrun devicectl device install app --device 02ECF10B-7023-54FA-BCDB-0243D3E676A1 /Users/lee/Library/Developer/Xcode/DerivedData/arface-acabocwrndbzhbgonmbvevpnvhwk/Build/Products/Debug-iphoneos/arface.app`.
- [x] Launch with `xcrun devicectl device process launch --device 02ECF10B-7023-54FA-BCDB-0243D3E676A1 com.leyi.arface`.
- [ ] User checks: outside spill, lip crease visibility, lower-lip highlight, `UV ML/prep/raw/dec/post/mask%`.
