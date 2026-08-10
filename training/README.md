# AR Face Makeup Parsing Training

This folder is the local training workspace for the lightweight face parsing
model used by the iOS AR makeup prototype.

## What This Trains

The current AR makeup stack is:

```text
ARKit tracking + full-face parsing + local refinement models + GPU makeup layers
```

The full-face model is intentionally small and practical:

- Input: `256 x 256` face ROI image
- Output: 9 segmentation classes
- Runtime target: iPhone 12, Core ML
- First dataset: LaPa

The 9 app classes are:

| id | name |
| --- | --- |
| 0 | background |
| 1 | skin |
| 2 | eyebrow |
| 3 | eye |
| 4 | nose |
| 5 | upper_lip |
| 6 | lower_lip |
| 7 | inner_mouth |
| 8 | hair |

LaPa's left/right eye and brow classes are merged because makeup rendering does
not need separate class ids for the first MVP.

The first local refinement model is the lip ROI model. It is trained on cropped
mouth regions and predicts only four classes:

| id | name |
| --- | --- |
| 0 | background_skin |
| 1 | upper_lip |
| 2 | lower_lip |
| 3 | inner_mouth |

This keeps lipstick edges more precise than asking the full-face model to solve
the mouth as a side task.

## One-Time Setup

Run these from the project root:

```bash
cd /Users/lee/CodeBuddy/leyi/arface
python3 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
pip install -r training/requirements.txt
```

If installing `torch` fails because of the Python version, install Python 3.11
or 3.12 and recreate the venv with that Python.

## Check LaPa

```bash
python training/prepare_lapa.py --root /Users/lee/Datasets/LaPa
```

You should see counts for `train`, `val`, and `test`, plus label ids from `0`
to `10`.

## Smoke Train

This is the first cheap check. It trains on only 200 images so we can confirm
the pipeline works before spending time.

```bash
python training/train.py \
  --root /Users/lee/Datasets/LaPa \
  --output-dir training/outputs/lapa_smoke \
  --limit-train 200 \
  --limit-val 80 \
  --epochs 3 \
  --batch-size 4
```

Expected output:

```text
training/outputs/lapa_smoke/best.pt
training/outputs/lapa_smoke/last.pt
```

## Longer Local Train

After the smoke run works:

```bash
python training/train.py \
  --root /Users/lee/Datasets/LaPa \
  --output-dir training/outputs/lapa_full_256 \
  --epochs 30 \
  --batch-size 8
```

If memory pressure is high on the M4 16GB Mac, use `--batch-size 4`.

If a run stops halfway, resume from the latest checkpoint and keep the same
total target epoch count:

```bash
python training/train.py \
  --root /Users/lee/Datasets/LaPa \
  --output-dir training/outputs/lapa_full_256 \
  --resume training/outputs/lapa_full_256/last.pt \
  --epochs 30 \
  --batch-size 4
```

The trainer writes `last.pt`, `best.pt`, and numbered snapshots such as
`epoch_005.pt` every few epochs.

To summarize a run while it is training:

```bash
python training/summarize_run.py \
  --run-dir training/outputs/lapa_full_256
```

## Evaluate

```bash
python training/evaluate.py \
  --root /Users/lee/Datasets/LaPa \
  --checkpoint training/outputs/lapa_smoke/best.pt
```

## Preview Masks

Before putting a checkpoint onto the phone, generate visual previews on the Mac:

```bash
python training/preview_masks.py \
  --root /Users/lee/Datasets/LaPa \
  --checkpoint training/outputs/lapa_mid_256/best.pt \
  --output-dir training/outputs/previews/lapa_mid_256 \
  --count 12
```

Each preview shows the source image, ground-truth lip mask, predicted lip mask,
and the raw predicted mask. Use this to catch obvious mistakes before testing on
iPhone.

## Lip ROI Refinement Model

Create the cropped lip dataset from LaPa:

```bash
python training/prepare_lip_dataset.py \
  --root /Users/lee/Datasets/LaPa \
  --output-root /Users/lee/Datasets/LaPaLip \
  --width 192 \
  --height 128
```

Smoke train the lip model:

```bash
python training/train_lip.py \
  --root /Users/lee/Datasets/LaPaLip \
  --output-dir training/outputs/lip_smoke \
  --limit-train 500 \
  --limit-val 120 \
  --epochs 5 \
  --batch-size 16
```

Longer local lip train:

```bash
python training/train_lip.py \
  --root /Users/lee/Datasets/LaPaLip \
  --output-dir training/outputs/lip_roi_192x128 \
  --epochs 40 \
  --batch-size 16
```

If memory pressure is high, use `--batch-size 8`.

Evaluate and preview before putting the model on the phone:

```bash
python training/evaluate_lip.py \
  --root /Users/lee/Datasets/LaPaLip \
  --checkpoint training/outputs/lip_roi_192x128/best.pt

python training/preview_lip_masks.py \
  --root /Users/lee/Datasets/LaPaLip \
  --checkpoint training/outputs/lip_roi_192x128/best.pt \
  --output-dir training/outputs/previews/lip_roi_192x128 \
  --count 12
```

Export the lip model to Core ML:

```bash
python training/export_coreml.py \
  --checkpoint training/outputs/lip_roi_192x128/best.pt \
  --output arface/Models/LipParsingSmall.mlpackage
```

If the phone reports high model latency, train a smaller realtime student. This
is the path for iPhone 12-class live preview; the larger lip model remains useful
as a quality baseline and future distillation target.

```bash
python training/prepare_lip_dataset.py \
  --root /Users/lee/Datasets/LaPa \
  --output-root /Users/lee/Datasets/LaPaLipRealtime \
  --width 128 \
  --height 96

python training/train_lip.py \
  --root /Users/lee/Datasets/LaPaLipRealtime \
  --output-dir training/outputs/lip_realtime_128x96_w16 \
  --image-width 128 \
  --image-height 96 \
  --model-width 16 \
  --epochs 40 \
  --batch-size 32 \
  --save-every 5

python training/evaluate_lip.py \
  --root /Users/lee/Datasets/LaPaLipRealtime \
  --checkpoint training/outputs/lip_realtime_128x96_w16/best.pt

python training/export_coreml.py \
  --checkpoint training/outputs/lip_realtime_128x96_w16/best.pt \
  --output arface/Models/LipParsingSmall.mlpackage
```

## Public Lip Parsing Datasets

LaPa is not enough for product-level lipstick edges, especially around mouth
corners, open-mouth frames, and the lower-lip outer contour. Use public parsing
datasets as a 2D lip-shape base, then use the ARKit/UV recordings for the final
app-specific student.

Convert CelebAMask-HQ class masks into the same 4-class lip ROI format:

```bash
python training/prepare_public_lip_dataset.py \
  --root /Users/lee/Datasets/CelebAMask-HQ \
  --output-root /Users/lee/Datasets/CelebAMaskHQLip \
  --image-dir CelebA-HQ-img \
  --mask-dir CelebAMask-HQ-mask-anno \
  --source-dataset celebamask_hq \
  --width 192 \
  --height 128 \
  --overwrite
```

The converter expects per-class mask filenames such as
`00000_u_lip.png`, `00000_l_lip.png`, and `00000_mouth.png`. For another
dataset with per-class masks, such as an iBugMask export, pass the matching
`--image-dir`, `--mask-dir`, and alias lists if needed.

Merge converted public datasets with LaPa lip crops:

```bash
python training/combine_lip_datasets.py \
  --sources lapa=/Users/lee/Datasets/LaPaLip,celeb=/Users/lee/Datasets/CelebAMaskHQLip \
  --output-root /Users/lee/Datasets/LipMixedPublic \
  --overwrite
```

Then train the 2D lip parser on the mixed set:

```bash
python training/train_lip.py \
  --root /Users/lee/Datasets/LipMixedPublic \
  --output-dir training/outputs/lip_mixed_public_192x128_w32 \
  --image-width 192 \
  --image-height 128 \
  --model-width 32 \
  --epochs 40 \
  --batch-size 16 \
  --save-every 5
```

## Export Core ML

```bash
mkdir -p arface/Models
python training/export_coreml.py \
  --checkpoint training/outputs/lapa_smoke/best.pt \
  --output arface/Models/MakeupParsingSmall.mlpackage
```

For the app, we will run the model every few frames and use landmarks to track
the mask between model frames. That is what keeps the final AR effect stable
without making the camera preview choppy.

## Cost Notes

This first path can run locally on the Mac and should not require paid cloud GPU.
App Store/TestFlight distribution later requires the Apple Developer Program.

LaPa is suitable for research/MVP validation. Before commercial launch, the
production model should be trained or fine-tuned on owned or licensed data.

## Teacher Path

The long-term model path is teacher/student:

```text
high-quality teacher masks -> small on-device student -> Core ML in the app
```

Keep teacher dependencies in a separate environment so the known-good training
and Core ML export environment stays stable:

```bash
deactivate
python3 -m venv .venv-teacher
source .venv-teacher/bin/activate
python -m pip install --upgrade pip
pip install -r training/requirements-teacher.txt
```

Start with FaRL/facer's LaPa face parser as an external teacher probe:

```bash
python training/farl_teacher_preview.py \
  --root /Users/lee/Datasets/LaPa \
  --split val \
  --output-dir training/outputs/teacher/farl_lapa448_preview \
  --count 24
```

This creates preview grids comparing the original LaPa lip labels with the FaRL
teacher output. Use it to validate whether the teacher improves upper-lip,
mouth-corner, and inner-mouth boundaries before generating larger pseudo-label
sets or distilling back into the mobile lip model.
