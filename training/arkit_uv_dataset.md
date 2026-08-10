# ARKit Lip UV Dataset Notes

This dataset is for training a 3D/UV lip student, not for runtime 2D overlay.

## Capture

Use the red record button in the ARKit app. Samples are written inside the app sandbox:

```text
Documents/ARKitLipUVRecordings/<yyyyMMdd-HHmmss>/
  session.json
  metadata.jsonl
  images/frame_000000.jpg
  frames/frame_000000.json
```

The recorder samples at 5 fps to keep device load manageable.

## Frame Schema

Each `frames/*.json` contains:

- `image`: raw `ARFrame.capturedImage` JPEG path plus pixel dimensions.
- `displayTransformPortrait`: ARKit display transform for mapping captured-image normalized coordinates to the portrait viewport.
- `cameraIntrinsics`, `cameraTransform`, `projectionMatrixPortrait`, `viewMatrixPortrait`.
- `faceTransform`: `ARFaceAnchor.transform`.
- `faceGeometry.vertices`: ARKit face-anchor-local vertices.
- `faceGeometry.textureCoordinates`: ARKit stable UV coordinates.
- `faceGeometry.triangleIndices`: mesh triangle indices.
- `blendShapes`: ARKit face blendshape values.
- `faceROI` and `lipROI`: portrait screen-space ROIs used by the app.

## Teacher-To-UV Flow

1. Export an `ARKitLipUVRecordings/<session>` folder from the iPhone app container.
2. Run a 2D teacher on each `images/*.jpg`.
3. Use `displayTransformPortrait` to align teacher mask coordinates with the saved portrait-space mesh projections.
4. Rasterize teacher lip mask into ARKit UV using `faceGeometry.textureCoordinates` and `triangleIndices`.
5. Save UV masks as the labels for the future student dataset.

Runtime should consume only UV/mesh outputs. The 2D teacher is used offline only.
