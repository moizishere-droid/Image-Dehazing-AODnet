"""Reproducible training, sealed test evaluation, figures and inference."""
import csv
import json
import math
import platform
import random
import statistics
from datetime import datetime, timezone
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .model import AODNet
from .metrics import scores
from .data import ROOT, load_split, sample_batch, sha256
from .checkpoint import save, load

def evaluate_cache(model, cache):
    model.eval(); rows = []
    with torch.inference_mode():
        for sid, h, c in cache:
            pred = model(h[None]).clamp(0,1)
            rows.append({"scene_id": sid, "input": scores(h[None], c[None]), "model": scores(pred, c[None])})
    return rows

def mean(rows, method, metric):
    return statistics.mean(r[method][metric] for r in rows)

def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    rng = random.Random(args.seed)
    train_data = load_split(args.manifest, "train", args.max_side)
    val_data = load_split(args.manifest, "validation", args.max_side)
    model = AODNet(); model.initialize_identity()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.05)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({"python":platform.python_version(), "torch":torch.__version__, "device":"cpu", "parameters":sum(p.numel() for p in model.parameters()), "manifest_sha256":sha256(args.manifest), "initialization":"from scratch; identity last layer", "selection":"highest mean validation PSNR", "training_scenes":len(train_data), "validation_scenes":len(val_data)})
    (args.output/"config.json").write_text(json.dumps(config,indent=2))
    best = mean(evaluate_cache(model,val_data),"model","psnr_db")
    metadata = {"config":config, "best_step":0, "validation_psnr_db":best}
    save(args.output/"best.pt",model,metadata)
    history=[]; start=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train(); h,c=sample_batch(train_data,rng,args.batch_size,args.patch)
        optimizer.zero_grad(); pred=model(h); loss=torch.mean((pred-c)**2)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step()
        if step%args.validate_every==0 or step==args.steps:
            rows=evaluate_cache(model,val_data)
            vp,vs=mean(rows,"model","psnr_db"),mean(rows,"model","ssim")
            record={"step":step,"training_batch_mse":loss.item(),"validation_psnr_db":vp,"validation_ssim":vs,"seconds":time.perf_counter()-start}
            history.append(record)
            if vp>best:
                best=vp;metadata={"config":config,"best_step":step,"validation_psnr_db":vp,"validation_ssim":vs};save(args.output/"best.pt",model,metadata)
            print(f"step {step}/{args.steps}: val PSNR {vp:.3f}, SSIM {vs:.4f}; best {best:.3f}",flush=True)
            (args.output/"history.json").write_text(json.dumps(history,indent=2))
    save(args.output/"last.pt",model,{"config":config,"step":args.steps})
    (args.output/"selection.json").write_text(json.dumps(metadata,indent=2))

def comparison(path, h, pred, clear, title):
    ims=[Image.fromarray(np.round(v.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)) for v in (h,pred,clear)]
    width=360; thumbs=[]
    for im in ims:
        im.thumbnail((width,300));thumbs.append(im)
    height=max(im.height for im in thumbs)+85
    canvas=Image.new("RGB",(width*3+40,height),"white");draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype("arial.ttf",15)
    except OSError:font=ImageFont.load_default()
    for i,(label,im) in enumerate(zip(("Hazy input","Model output","Clear reference"),thumbs)):
        x=10+i*(width+10);draw.text((x,10),label,fill="black",font=font);canvas.paste(im,(x,35))
    draw.text((10,height-30),title,fill="black",font=font)
    Path(path).parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def evaluate(args):
    torch.set_num_threads(args.threads)
    model,metadata=load(args.checkpoint)
    if metadata.get("config",{}).get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("Checkpoint and split manifest do not match")
    cache=load_split(args.manifest,"test",args.max_side);rows=evaluate_cache(model,cache)
    args.output.mkdir(parents=True,exist_ok=True)
    summary={"input":{},"model":{}}
    for method in summary:
        for metric in ("psnr_db","ssim"):summary[method][metric]=mean(rows,method,metric)
    result={"evaluated_at_utc":datetime.now(timezone.utc).isoformat(), "dataset":"O-HAZE", "test_scenes":len(rows), "protocol":"Custom scene-level 35 train / 5 validation / 5 test split; all test scenes unseen by this from-scratch model", "max_side":args.max_side,"resize":"aspect-ratio preserving Lanczos", "data_range":1, "color":"RGB", "psnr":"per-image RGB MSE then arithmetic mean of PSNR; no crop", "ssim":"11x11 Gaussian, sigma=1.5, population covariance, K1=.01, K2=.03; valid window (5px boundary excluded)","checkpoint_sha256":sha256(args.checkpoint),"manifest_sha256":sha256(args.manifest),"metadata":metadata,"summary":summary,"per_scene":rows}
    (args.output/"test_results.json").write_text(json.dumps(result,indent=2))
    with (args.output/"per_scene.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["scene_id","input_psnr_db","model_psnr_db","input_ssim","model_ssim"])
        for r in rows:writer.writerow([r["scene_id"],r["input"]["psnr_db"],r["model"]["psnr_db"],r["input"]["ssim"],r["model"]["ssim"]])
    # Show every held-out test scene, in ID order, instead of selecting the best-looking example.
    with torch.inference_mode():
        for sid,h,c in sorted(cache):
            pred=model(h[None])[0].clamp(0,1)
            row=next(r for r in rows if r["scene_id"]==sid)
            title=f"Test scene {sid} | PSNR {row['input']['psnr_db']:.2f} -> {row['model']['psnr_db']:.2f} dB | SSIM {row['model']['ssim']:.4f}"
            comparison(args.output/"figures"/f"test_{sid}.png",h,pred,c,title)
    print(json.dumps(summary,indent=2),flush=True)

def infer(args):
    torch.set_num_threads(args.threads);model,_=load(args.checkpoint)
    with Image.open(args.input) as im:
        im=im.convert("RGB")
        if args.max_side and max(im.size)>args.max_side:
            scale=args.max_side/max(im.size);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
        x=torch.from_numpy((np.asarray(im).copy().astype(np.float32)/255).transpose(2,0,1))
    with torch.inference_mode():pred=model(x[None])[0].clamp(0,1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.round(pred.permute(1,2,0).numpy()*255).astype(np.uint8)).save(args.output)
    print(f"Saved {args.output}")
