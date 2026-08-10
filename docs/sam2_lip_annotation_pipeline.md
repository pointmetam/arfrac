# SAM2 Lip Annotation Pipeline

This pipeline replaces noisy face-parser teacher labels with cleaner video-assisted lip masks.
For CVAT setup and export layout, see `docs/cvat_lip_annotation.md`.

## 1. Prepare Portrait Frames

```bash
.venv-training/bin/python training/prepare_sam_lip_annotation.py \
  --session /private/tmp/arface-uv-recordings-new/20260528-180100 \
  --output-dir training/outputs/sam2_lip_annotation_20260528-180100 \
  --stride 4 \
  --max-frames 80 \
  --overwrite
```

Annotate `images/*.jpg` in SAM2/CVAT/SAMannot. These images are portrait-oriented and should be exported in portrait mask space.

Expected mask folders:

```text
masks/upper_lip/frame_000000.png
masks/lower_lip/frame_000000.png
masks/inner_mouth/frame_000000.png
```

## 2. Combine Class Masks

```bash
.venv-training/bin/python training/combine_sam_lip_masks.py \
  --input-dir training/outputs/sam2_lip_annotation_20260528-180100/masks \
  --output-dir training/outputs/sam2_lip_annotation_20260528-180100/combined_masks \
  --image-dir training/outputs/sam2_lip_annotation_20260528-180100/images \
  --preview-dir training/outputs/sam2_lip_annotation_20260528-180100/combined_previews
```

The output labels use:

- `0`: background
- `1`: upper lip
- `2`: lower lip
- `3`: inner mouth

## 3. Project 2D SAM Masks To ARKit UV

```bash
.venv-training/bin/python training/teacher_2d_to_arkit_uv.py \
  --session /private/tmp/arface-uv-recordings-new/20260528-180100 \
  --output-dir training/outputs/sam2_uv_20260528-180100 \
  --mask-dir training/outputs/sam2_lip_annotation_20260528-180100/combined_masks \
  --mask-space portrait \
  --uv-size 512 \
  --count 999 \
  --preview-count 24 \
  --cull-backfaces
```

Check `training/outputs/sam2_uv_20260528-180100/previews` before training.

## 4. Build A UV Teacher Dataset

After each annotated session has a `training/outputs/sam2_uv_<session>/uv_labels` folder:

```bash
.venv-training/bin/python training/generate_arkit_uv_teacher_dataset.py \
  --sessions-root /private/tmp/arface-uv-recordings-new \
  --output-root /Users/lee/Datasets/ARKitLipUVTeacherSAM2 \
  --label-root-template 'training/outputs/sam2_uv_{session}/uv_labels' \
  --metrics-template 'training/outputs/sam2_uv_{session}/metrics.csv' \
  --session-prefix 20260528- \
  --overwrite
```

## 5. Train UV 4C Student

```bash
.venv-training/bin/python training/train_arkit_uv_student.py \
  --root /Users/lee/Datasets/ARKitLipUVTeacherSAM2 \
  --output-dir training/outputs/arkit_uv_student_sam2_4c_128x96_w24_e40 \
  --uv-size 128 \
  --image-width 128 \
  --image-height 96 \
  --crop-box 32,64,92,108 \
  --model-width 24 \
  --epochs 40 \
  --batch-size 16 \
  --class-weights 1.0,4.0,4.0,0.5 \
  --best-metric lip_union_iou \
  --cache-uv-images \
  --save-every 10
```
