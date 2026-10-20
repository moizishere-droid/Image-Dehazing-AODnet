"""Scene-level pairing and deterministic data manifests."""
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_manifest(dataset, output, seed=2026):
    dataset, output = Path(dataset).resolve(), Path(output)
    groups = {}
    for kind in ("hazy", "clear"):
        mapping = {}
        for p in sorted((dataset / kind).iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
            match = re.match(r"(\d+)_", p.name)
            if not match: raise ValueError(f"Cannot determine scene ID: {p.name}")
            key = match.group(1)
            if key in mapping: raise ValueError(f"Duplicate scene {key}")
            mapping[key] = p
        groups[kind] = mapping
    if set(groups["hazy"]) != set(groups["clear"]):
        raise ValueError("Missing matched references")
    ids = sorted(groups["hazy"])
    if len(ids) != 45: raise ValueError(f"Expected all 45 O-HAZE scenes, got {len(ids)}")
    random.Random(seed).shuffle(ids)
    rows = []
    for i, key in enumerate(ids):
        split = "train" if i < 35 else ("validation" if i < 40 else "test")
        row = {"scene_id": key, "split": split}
        for kind in ("hazy", "clear"):
            p = groups[kind][key]
            row[kind] = p.relative_to(ROOT).as_posix()
            row[kind+"_sha256"] = sha256(p)
        rows.append(row)
    result = {"dataset": "O-HAZE", "seed": seed, "protocol": "Custom 35/5/5 scene split; not the official challenge split", "rows": rows}
    validate_manifest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate_manifest(manifest):
    rows=manifest['rows']
    ids=[r['scene_id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Scene leakage: repeated scene ID')
    for kind in ('clear_sha256','hazy_sha256'):
        seen={}
        for row in rows:
            key=row[kind]
            if key in seen and seen[key]!=row['split']: raise ValueError('Image content repeated across splits')
            seen[key]=row['split']
    counts={k:sum(r['split']==k for r in rows) for k in ('train','validation','test')}
    if counts!={'train':35,'validation':5,'test':5}: raise ValueError('Expected 35/5/5 scene split')
    return manifest

def load_split(manifest, split, max_side=384):
    rows = validate_manifest(json.loads(Path(manifest).read_text(encoding="utf-8")))["rows"]
    selected, cache = [r for r in rows if r["split"] == split], []
    for row in selected:
        ims = []
        for kind in ("hazy", "clear"):
            path = ROOT / row[kind]
            if sha256(path) != row[kind+"_sha256"]: raise ValueError(f"Changed input: {path}")
            with Image.open(path) as im: ims.append(ImageOps.exif_transpose(im).convert("RGB"))
        if ims[0].size != ims[1].size: raise ValueError("Paired dimensions differ")
        w, h = ims[0].size; scale = min(1., max_side / max(w, h))
        size = (round(w*scale), round(h*scale))
        tensors = []
        for im in ims:
            im = im.resize(size, Image.Resampling.LANCZOS)
            tensors.append(torch.from_numpy((np.asarray(im).copy().astype(np.float32)/255).transpose(2,0,1)))
        cache.append((row["scene_id"], *tensors))
    if not cache: raise ValueError(f"Empty {split} split")
    return cache

def sample_batch(cache, rng, batch_size, patch):
    hs, cs = [], []
    for _ in range(batch_size):
        _, h, c = cache[rng.randrange(len(cache))]
        height, width = h.shape[-2:]
        if min(height, width) < patch: raise ValueError("Image is smaller than patch")
        y, x = rng.randrange(height-patch+1), rng.randrange(width-patch+1)
        h, c = h[:,y:y+patch,x:x+patch], c[:,y:y+patch,x:x+patch]
        if rng.random() < .5: h, c = h.flip(-1), c.flip(-1)
        if rng.random() < .5: h, c = h.flip(-2), c.flip(-2)
        hs.append(h); cs.append(c)
    return torch.stack(hs), torch.stack(cs)
