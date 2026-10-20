"""Build the README, evaluation report and figures from recorded experiment outputs."""
from pathlib import Path
import sys,json,statistics
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dehaze.checkpoint import load
from dehaze.metrics import scores
from dehaze.experiment import comparison

r=json.loads((ROOT/'reports/test_results.json').read_text())
history=json.loads((ROOT/'artifacts/aodnet/history.json').read_text())
split=json.loads((ROOT/'reports/splits.json').read_text())
selection=json.loads((ROOT/'artifacts/aodnet/selection.json').read_text())
evaluation_date=datetime.fromisoformat(r['evaluated_at_utc']).strftime('%d %B %Y')
figdir=ROOT/'reports/figures';figdir.mkdir(exist_ok=True)
a,b=r['summary']['input'],r['summary']['model']
delta=b['psnr_db']-a['psnr_db'];ds=b['ssim']-a['ssim']
# Training batch loss is intentionally labelled as a batch measure, not an epoch mean.
fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
steps=[h['step'] for h in history]
axes[0].plot(steps,[h['training_batch_mse'] for h in history],color='#087c94',marker='o',markersize=3)
axes[0].set(title='Training batch loss',xlabel='Optimizer step',ylabel='MSE (sampled batch)')
axes[1].plot(steps,[h['validation_psnr_db'] for h in history],color='#5e44a8',marker='o',markersize=3)
axes[1].axvline(selection['best_step'],color='#777777',linestyle='--',label=f"Selected step {selection['best_step']}")
axes[1].set(title='Validation checkpoint selection',xlabel='Optimizer step',ylabel='Mean PSNR (dB)');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(figdir/'training.png',dpi=180);plt.close(fig)
# An input / model / clear-reference figure for every reserved test scene.
imgs=[Image.open(figdir/f"test_{row['scene_id']}.png").convert('RGB') for row in sorted(r['per_scene'],key=lambda x:x['scene_id'])]
gallery=Image.new('RGB',(max(im.width for im in imgs),sum(im.height for im in imgs)),'white');top=0
for im in imgs:gallery.paste(im,(0,top));top+=im.height
gallery.save(figdir/'test_gallery.png')
# Reproduce the bundled indoor comparison, explicitly outside the O-HAZE average.
model,_=load(ROOT/'artifacts/aodnet/best.pt');torch.set_num_threads(2)
pair=[]
for name in ('hazy','clear'):
 with Image.open(ROOT/f'assets/samples/{name}.jpg') as im:
  im=im.convert('RGB');pair.append(torch.from_numpy((np.asarray(im).copy().astype(np.float32)/255).transpose(2,0,1)))
with torch.inference_mode():pred=model(pair[0][None])[0].clamp(0,1)
sample={'input':scores(pair[0][None],pair[1][None]),'model':scores(pred[None],pair[1][None]),'scope':'One bundled indoor example; not part of the O-HAZE training or held-out average'}
(ROOT/'reports/bundled_sample.json').write_text(json.dumps(sample,indent=2))
comparison(figdir/'bundled_sample.png',pair[0],pred,pair[1],'Bundled indoor example: qualitative illustration, excluded from test averages')
test_info=json.loads((ROOT/'reports/tests.json').read_text())
perrows='\n'.join(f"| {x['scene_id']} | {x['input']['psnr_db']:.2f} | {x['model']['psnr_db']:.2f} | {x['input']['ssim']:.4f} | {x['model']['ssim']:.4f} |" for x in sorted(r['per_scene'],key=lambda q:q['scene_id']))
train_ids=', '.join(x['scene_id'] for x in split['rows'] if x['split']=='train')
val_ids=', '.join(x['scene_id'] for x in split['rows'] if x['split']=='validation')
test_ids=', '.join(x['scene_id'] for x in split['rows'] if x['split']=='test')
readme=f"""# Image Dehazing with AOD-Net

A compact PyTorch project that restores haze-degraded images and measures the result against a matched clear reference. It includes training, validation-based checkpoint selection, image inference, PSNR/SSIM evaluation, automated tests, and visual comparisons.

**Model:** 5-layer AOD-Net | **Parameters:** 1,761 | **Runtime:** CPU | **Evaluation date:** {evaluation_date}

## Results

Measured on **5 held-out O-HAZE scenes**, using a custom **35 training / 5 validation / 5 test** scene split. Images preserve their aspect ratio and are resized to a maximum side of **384 pixels**.

| Mean RGB metric (higher is better) | Hazy input | Model output | Difference |
|---|---:|---:|---:|
| PSNR | {a['psnr_db']:.2f} dB | **{b['psnr_db']:.2f} dB** | **+{delta:.2f} dB** |
| SSIM | {a['ssim']:.4f} | **{b['ssim']:.4f}** | **+{ds:.4f}** |

The model improved both metrics on all five test scenes. These are small-scale educational results, **not an official O-HAZE challenge score**. The test scenes were excluded from this model's training and checkpoint selection. The network was trained from scratch; no pretrained weights were used.

[Per-scene results](reports/per_scene.csv) | [Full evaluation protocol](reports/evaluation.md) | [Machine-readable results](reports/test_results.json)

## Image comparisons

All five test scenes are shown below, in scene-ID order. Each row contains **hazy input -> model output -> clear reference**. No test examples are omitted.

![All five held-out scene comparisons](reports/figures/test_gallery.png)

### Bundled indoor example

This is the same bundled hazy/clear scene supplied with the academic project, processed by the included model. It is a separate qualitative example and does not contribute to the O-HAZE averages. Performance varies with scene and haze conditions; [its individual scores](reports/bundled_sample.json) are included.

![Bundled indoor image comparison](reports/figures/bundled_sample.png)

## How it works

1. Pair each hazy image with its clear reference and assign complete scenes to a fixed split.
2. Normalize RGB values to [0, 1]. Resize without stretching; sample paired 96 x 96 training crops with matching flips.
3. Train the 5-layer AOD-Net using mean squared error, Adam and a cosine learning-rate schedule.
4. Select the saved checkpoint with the highest mean validation PSNR.
5. Evaluate that checkpoint on the reserved test scenes and export per-scene metrics and comparison figures.

```mermaid
flowchart LR
  H[Hazy RGB image] --> P[Normalize and preserve aspect ratio]
  P --> M[5-layer AOD-Net]
  M --> O[Restored image]
  O --> E[PSNR / SSIM against clear reference]
```

## Training and evaluation

The recorded run used **600 optimizer steps**, batch size **8**, crop size **96**, initial learning rate **0.001**, random seed **2026**, and **2 CPU threads**. Validation ran every 50 steps. The selected checkpoint is step **{selection['best_step']}**, chosen before test evaluation. Later steps did not improve validation PSNR, so the last checkpoint was not used for reporting.

![Training loss and validation checkpoint selection](reports/figures/training.png)

The left plot shows sampled training-batch MSE, not an epoch average. Logs, settings, checkpoint selection and file hashes are retained under [artifacts/aodnet](artifacts/aodnet) and [reports](reports).

## Reproduce the results

Run these commands from the repository root using **Python 3.12**. A GPU is not required. The official dataset download is approximately **547 MiB**; raw dataset files are ignored by Git.

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-report.txt

# Download the official paired images and verify archive CRCs.
python scripts/download_data.py

# Create the fixed 35/5/5 scene split, with per-image SHA-256 hashes.
python -m dehaze prepare --seed 2026

# Reproduce training and validation-based checkpoint selection.
python -m dehaze train --steps 600 --seed 2026 --batch-size 8 --patch 96 --max-side 384 --validate-every 50 --threads 2

# Evaluate the selected checkpoint and regenerate visual reports.
python -m dehaze evaluate --max-side 384 --threads 2
python scripts/check.py
python scripts/report.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. To evaluate the supplied checkpoint without retraining, complete setup, download the dataset, and run only `prepare`, `evaluate`, and `scripts/report.py`. Small floating-point differences can occur across platforms or library builds.

### Try an image

```powershell
python -m dehaze infer --input assets/samples/hazy.jpg --output outputs/restored.png
```

### Run the tests

```powershell
python -m unittest discover -s tests -v
```

**Local result: {test_info['tests_run']} tests passed**, covering scene separation, missing pairs, paired augmentation, known metric values, checkpoint round trips, model shape/parameter checks, and a real optimizer update. [Recorded test run](reports/tests.txt). GitHub Actions is configured to run the tests on pushes and pull requests; its remote status is separate from this local result.

## Dataset and metric conventions

- **Dataset:** [O-HAZE, Ancuti et al., CVPR Workshops 2018](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/), 45 real outdoor hazy/clear scene pairs.
- **PSNR:** computed from per-image RGB mean squared error over values in [0, 1], then averaged across test scenes. No image crop is used for PSNR.
- **SSIM:** 11 x 11 Gaussian window, sigma 1.5, population covariance, K1=0.01, K2=0.03, RGB channel average. The valid window excludes a 5-pixel boundary.
- Both input and output are compared with the same resized clear reference. The output is clipped to [0, 1].
- This project uses a custom scene split and reduced resolution. Compare numbers only when dataset, split, resolution and metric definitions match.

## Scope

This is a small supervised academic/portfolio experiment, not a production image-enhancement service or a new dehazing architecture. Five held-out scenes do not establish general performance. The model can alter brightness, color or detail, and the indoor example is outside the outdoor training distribution.

## Repository layout

```text
dehaze/                 Model, data, metrics, training, evaluation and inference
scripts/                Dataset download and report generation
tests/                  Automated checks
artifacts/aodnet/       Selected checkpoint, configuration and training history
reports/                Split manifest, per-scene results, tests and figures
assets/samples/         Bundled example images
datasets/               Downloaded official data (not committed)
```

## Attribution

Architecture: **Boyi Li et al., AOD-Net: All-in-One Dehazing Network (ICCV 2017)**. Implementation reference: [walsvid/AOD-Net-PyTorch](https://github.com/walsvid/AOD-Net-PyTorch). This project applies and evaluates an existing architecture; it does not claim architectural novelty.

Dataset citation: Codruta O. Ancuti, Cosmin Ancuti, Radu Timofte and Christophe De Vleeschouwer, **O-HAZE: A Dehazing Benchmark with Real Hazy and Haze-Free Outdoor Images**, CVPR Workshops 2018. The dataset remains attributed to its creators and is downloaded from their official host.
"""
(ROOT/'README.md').write_text(readme,encoding='utf-8')
evaluation=f"""# Evaluation report

Evaluation date: **{evaluation_date}**. Training starts from random weights, with the final layer initialized to the identity mapping. Checkpoint selection uses validation only. The test labels are used only for the final evaluation, after training and selection.

## Split and provenance

- Official O-HAZE source: https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/
- Archive: https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip
- Train (35): {train_ids}
- Validation (5): {val_ids}
- Test (5): {test_ids}
- Seed: 2026. Scene IDs are disjoint; no patches from a scene are shared across splits.
- This is a custom 35/5/5 split, not the official NTIRE challenge split.
- Per-file SHA-256 digests and assignments: [splits.json](splits.json). Downloaded files are additionally verified using archive CRC values.

## Method

AOD-Net has five convolutional layers and 1,761 trainable parameters. RGB inputs are normalized to [0,1]. Images are resized with Lanczos to maximum side 384 while preserving aspect ratio. Training samples 96 x 96 paired crops and applies matching horizontal/vertical flips. Adam uses learning rate 0.001, cosine decay to 0.00005, MSE loss, gradient clipping at norm 1, batch size 8, and 600 steps. No test-driven parameter tuning was performed.

Validation occurs every 50 steps; the best mean validation PSNR chooses the saved checkpoint. Selected step: **{selection['best_step']}**. Validation PSNR: **{selection['validation_psnr_db']:.4f} dB**. CPU PyTorch: **{selection['config']['torch']}**, Python **{selection['config']['python']}**. Core dependencies are pinned in requirements.txt.

## Held-out results

| Scene | Input PSNR (dB) | Model PSNR (dB) | Input SSIM | Model SSIM |
|---|---:|---:|---:|---:|
{perrows}
| **Mean** | **{a['psnr_db']:.4f}** | **{b['psnr_db']:.4f}** | **{a['ssim']:.4f}** | **{b['ssim']:.4f}** |

Mean PSNR difference: **+{delta:.4f} dB**. Mean SSIM difference: **+{ds:.4f}**. Both metrics improve on every reserved scene. Raw values: [test_results.json](test_results.json). All five comparisons: [gallery](figures/test_gallery.png).

## Definitions

PSNR uses per-image RGB MSE and a data range of 1, with no cropping. The table reports the arithmetic mean of per-image PSNR, not PSNR from pooled MSE. SSIM uses a normalized 11 x 11 Gaussian window with sigma 1.5, population covariance, K1=.01, K2=.03 and the valid window, averaging RGB channels. Output is clipped before evaluation. Pixel alignment and resizing are identical for each matched pair.

## Limits

Only five held-out scenes and one random split were evaluated. These are reduced-resolution educational results and cannot be compared directly with published O-HAZE benchmarks using different splits, resolutions or definitions. No real-time, deployment-quality or broad generalization claim is made. The separate bundled indoor example is not included in test averages. It has PSNR {sample['model']['psnr_db']:.4f} dB and SSIM {sample['model']['ssim']:.4f}; the input baseline is {sample['input']['psnr_db']:.4f} dB and {sample['input']['ssim']:.4f}.

## Reproduction evidence

- [Training configuration](../artifacts/aodnet/config.json)
- [Training history](../artifacts/aodnet/history.json)
- [Checkpoint selection](../artifacts/aodnet/selection.json)
- [Split and image hashes](splits.json)
- [Per-scene CSV](per_scene.csv)
- [Test run](tests.txt)
- [Machine-readable metrics and checkpoint hash](test_results.json)

The local tests passed; the GitHub workflow has not been executed by this local run. Files can be pushed without committing datasets, virtual environments or generated inference outputs.
"""
(ROOT/'reports/evaluation.md').write_text(evaluation,encoding='utf-8')
print('README, detailed evaluation, training chart, full test gallery and bundled example generated.')
