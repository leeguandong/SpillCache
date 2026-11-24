# SpillCache: controlled trajectory-replay study

Reproduction material for **SpillCache: Deviation-Space Reconstruction for Selective Token Computation in Diffusion Transformer Image Editing**, by Guandong Li and Mengxia Ye.

This artifact studies **which representation to interpolate** at a fixed active-token budget. It compares absolute edit velocities with scheduler-index-aligned edit-minus-no-op deviations. It is not a closed-loop sparse inference implementation or a measured acceleration claim.

## Reproduce the reported aggregates without a GPU

From this directory, using Python 3.10 or newer:

```bash
python scripts/summarize_lift_quality.py --root results/quality --output quality_summary.json
python scripts/summarize_lifting_fidelity.py --root results/fidelity --exclude-last-step --output fidelity_summary.json
```

The package includes all 24 per-sample numeric records at **K=0.25**. The `checkerboard` row uses identical regular active positions and an identical nearest-neighbor/Gaussian operator for both representations. It reproduces LPIPS **0.7719 -> 0.1817**, PSNR **12.09 -> 23.83 dB**, and SSIM **0.207 -> 0.832**. The `edit` rows are oracle magnitude selections from captured dense edit/no-op differences; they are not deployable online selectors.

The pilot is manually selected (8 add, 4 attribute, 4 remove, 8 replace). Original sample IDs and instructions are in `data/pilot_manifest.jsonl`; the 24 samples are not an official benchmark split. Numeric records preserve the original results and omit machine-specific metadata. Excluding the final step in the velocity aggregate matches the paper's steps 0–26 comparison.

## Capture and replay on a GPU

Obtain Qwen-Image-Edit-2511 and a compatible `QwenImageEditPlusPipeline` implementation under their own terms. The original experiments recorded a local diffusers checkout at commit `97c2c6e3971243a081dc879496166fc8848baca5`; that third-party checkout is not bundled here. Install the GPU dependencies in `requirements-gpu.txt` and supply the checkout and model locations explicitly.

Get the original source images from [ImgEdit-Bench](https://github.com/PKU-YuanGroup/ImgEdit). Place the source images at the paths declared in `data/pilot_manifest.jsonl`, or adjust those paths to an existing local benchmark copy. Images, pretrained weights, and third-party source code are not redistributed.

```bash
python -m scripts.run_lifting_fidelity --sample-id att_065 --model-path /path/to/Qwen-Image-Edit-2511 --diffusers-path /path/to/diffusers --budgets 0.25 --output outputs/lifting_fidelity/att_065
python -m scripts.run_endtoend_lift_quality --sample-id att_065 --model-path /path/to/Qwen-Image-Edit-2511 --diffusers-path /path/to/diffusers --budgets 0.25 --dense-only-last 1 --output outputs/lift_quality/att_065
```

The historical filename `run_endtoend_lift_quality.py` refers to decoding the final replay latent to an image. Its velocities are captured from dense trajectories, not reevaluated by a sparse transformer at the replayed states. The final scheduler step is dense. The original protocol uses 28 steps, true CFG 4.0, seed 42, and 1024×1024 images.

## Scope and provenance

The interpolation and replay functions are copied from the research snapshot. Changes in this artifact are limited to explanatory module docstrings, portable default paths, a default budget of 0.25, and packaging. Original source hashes are in `PROVENANCE.json`. No new GPU experiments were performed for this package; the existing per-sample numeric aggregates were rerun on CPU and checked. No sparse backend, relation selector, GraphLifter, or system speedup is claimed by these results.

This package contains the authors' reproduction scripts and numeric records. Third-party benchmarks, models, and libraries retain their own licenses. No additional software license grant is made by this snapshot.
