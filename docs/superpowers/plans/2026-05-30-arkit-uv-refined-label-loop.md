# ARKit UV Refined Label Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a small, safe loop for refining ARKit UV lip labels without damaging the original teacher dataset.

**Architecture:** Keep `/Users/lee/Datasets/ARKitLipUVTeacher` immutable. Create a copied refined dataset with the same train/val/test structure, then generate a ranked review manifest and visual previews for the frames most likely to need manual label correction.

**Tech Stack:** Python, PIL, JSONL metadata, existing ARKit UV projection helpers and unittest tests.

---

### Task 1: Refined Dataset Bootstrap

**Files:**
- Create: `/Users/lee/CodeBuddy/leyi/arface/training/prepare_arkit_uv_refined_dataset.py`
- Test: `/Users/lee/CodeBuddy/leyi/arface/training/tests/test_arkit_uv_refined_dataset.py`

- [ ] Write tests for copying split folders, metadata, summary, and refusing to overwrite an existing output.
- [ ] Implement `prepare_refined_dataset(source_root, output_root, splits)`.
- [ ] Add CLI args for `--source-root`, `--output-root`, and `--splits`.

### Task 2: Hard Frame Selector

**Files:**
- Create: `/Users/lee/CodeBuddy/leyi/arface/training/select_arkit_uv_hard_frames.py`
- Test: `/Users/lee/CodeBuddy/leyi/arface/training/tests/test_arkit_uv_refined_dataset.py`

- [ ] Write tests for ranking frames by metadata risk: low previous UV IoU, centroid jump, yaw, mouth opening, and lip-area extremes.
- [ ] Implement candidate scoring with reason strings.
- [ ] Write a JSONL manifest and portrait/UV preview images.

### Task 3: Verify Locally

**Files:**
- Verify: `/Users/lee/CodeBuddy/leyi/arface/training/tests/test_arkit_uv_refined_dataset.py`

- [ ] Run `python -m unittest training.tests.test_arkit_uv_refined_dataset`.
- [ ] Run `python -m unittest discover -s training/tests`.
- [ ] Smoke-run hard-frame selection on a tiny count against the current ARKit UV teacher dataset.
