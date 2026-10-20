# Evaluation report

Evaluation date: **20 October 2026**. Training starts from random weights, with the final layer initialized to the identity mapping. Checkpoint selection uses validation only. The test labels are used only for the final evaluation, after training and selection.

## Split and provenance

- Official O-HAZE source: https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/
- Archive: https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip
- Train (35): 26, 24, 45, 25, 30, 14, 05, 28, 06, 02, 44, 18, 23, 31, 19, 17, 12, 09, 13, 41, 11, 42, 22, 35, 40, 34, 37, 10, 04, 03, 20, 01, 16, 29, 32
- Validation (5): 38, 27, 36, 39, 15
- Test (5): 07, 43, 33, 21, 08
- Seed: 2026. Scene IDs are disjoint; no patches from a scene are shared across splits.
- This is a custom 35/5/5 split, not the official NTIRE challenge split.
- Per-file SHA-256 digests and assignments: [splits.json](splits.json). Downloaded files are additionally verified using archive CRC values.

## Method

AOD-Net has five convolutional layers and 1,761 trainable parameters. RGB inputs are normalized to [0,1]. Images are resized with Lanczos to maximum side 384 while preserving aspect ratio. Training samples 96 x 96 paired crops and applies matching horizontal/vertical flips. Adam uses learning rate 0.001, cosine decay to 0.00005, MSE loss, gradient clipping at norm 1, batch size 8, and 600 steps. No test-driven parameter tuning was performed.

Validation occurs every 50 steps; the best mean validation PSNR chooses the saved checkpoint. Selected step: **50**. Validation PSNR: **17.4900 dB**. CPU PyTorch: **2.14.1+cpu**, Python **3.12.14**. Core dependencies are pinned in requirements.txt.

## Held-out results

| Scene | Input PSNR (dB) | Model PSNR (dB) | Input SSIM | Model SSIM |
|---|---:|---:|---:|---:|
| 07 | 17.34 | 21.72 | 0.7929 | 0.8041 |
| 08 | 11.27 | 14.09 | 0.5119 | 0.5412 |
| 21 | 10.59 | 15.63 | 0.5452 | 0.6690 |
| 33 | 14.90 | 18.61 | 0.7008 | 0.8141 |
| 43 | 16.55 | 19.59 | 0.7739 | 0.8557 |
| **Mean** | **14.1295** | **17.9280** | **0.6650** | **0.7368** |

Mean PSNR difference: **+3.7986 dB**. Mean SSIM difference: **+0.0719**. Both metrics improve on every reserved scene. Raw values: [test_results.json](test_results.json). All five comparisons: [gallery](figures/test_gallery.png).

## Definitions

PSNR uses per-image RGB MSE and a data range of 1, with no cropping. The table reports the arithmetic mean of per-image PSNR, not PSNR from pooled MSE. SSIM uses a normalized 11 x 11 Gaussian window with sigma 1.5, population covariance, K1=.01, K2=.03 and the valid window, averaging RGB channels. Output is clipped before evaluation. Pixel alignment and resizing are identical for each matched pair.

## Limits

Only five held-out scenes and one random split were evaluated. These are reduced-resolution educational results and cannot be compared directly with published O-HAZE benchmarks using different splits, resolutions or definitions. No real-time, deployment-quality or broad generalization claim is made. The separate bundled indoor example is not included in test averages. It has PSNR 18.3467 dB and SSIM 0.8447; the input baseline is 25.8236 dB and 0.9294.

## Reproduction evidence

- [Training configuration](../artifacts/aodnet/config.json)
- [Training history](../artifacts/aodnet/history.json)
- [Checkpoint selection](../artifacts/aodnet/selection.json)
- [Split and image hashes](splits.json)
- [Per-scene CSV](per_scene.csv)
- [Test run](tests.txt)
- [Machine-readable metrics and checkpoint hash](test_results.json)

The local tests passed; the GitHub workflow has not been executed by this local run. Files can be pushed without committing datasets, virtual environments or generated inference outputs.
