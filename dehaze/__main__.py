import argparse
from pathlib import Path
from .data import build_manifest
from .experiment import train, evaluate, infer

def main():
    p=argparse.ArgumentParser(description="AOD-Net: reproducible image dehazing")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("prepare");d.add_argument("--dataset",type=Path,default=Path("datasets/o_haze"));d.add_argument("--output",type=Path,default=Path("reports/splits.json"));d.add_argument("--seed",type=int,default=2026)
    t=sub.add_parser("train");t.add_argument("--manifest",type=Path,default=Path("reports/splits.json"));t.add_argument("--output",type=Path,default=Path("artifacts/aodnet"));t.add_argument("--seed",type=int,default=2026);t.add_argument("--steps",type=int,default=600);t.add_argument("--batch-size",type=int,default=8);t.add_argument("--patch",type=int,default=96);t.add_argument("--lr",type=float,default=.001);t.add_argument("--max-side",type=int,default=384);t.add_argument("--validate-every",type=int,default=50);t.add_argument("--threads",type=int,default=2)
    e=sub.add_parser("evaluate");e.add_argument("--manifest",type=Path,default=Path("reports/splits.json"));e.add_argument("--checkpoint",type=Path,default=Path("artifacts/aodnet/best.pt"));e.add_argument("--output",type=Path,default=Path("reports"));e.add_argument("--max-side",type=int,default=384);e.add_argument("--threads",type=int,default=2)
    i=sub.add_parser("infer");i.add_argument("--checkpoint",type=Path,default=Path("artifacts/aodnet/best.pt"));i.add_argument("--input",type=Path,required=True);i.add_argument("--output",type=Path,required=True);i.add_argument("--max-side",type=int,default=384);i.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if args.command=="prepare":
        data=build_manifest(args.dataset,args.output,args.seed);print(f"Wrote {len(data['rows'])} scenes to {args.output}")
    elif args.command=="train":
        if min(args.steps,args.batch_size,args.patch,args.validate_every,args.threads)<=0:p.error("Training sizes and intervals must be positive")
        train(args)
    elif args.command=="evaluate":evaluate(args)
    elif args.command=="infer":infer(args)
if __name__=="__main__":main()
