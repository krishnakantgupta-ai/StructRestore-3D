#!/usr/bin/env python3
"""StructRestore-3D baseline: masked-pixel self-supervised denoising smoke test.
This is a development baseline, not a validated new method.
"""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,width,3,padding=1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
            nn.Conv2d(width,1,3,padding=1))
    def forward(self,x): return self.net(x)

def mask_replace(x, fraction=0.05):
    # Randomly replace selected pixels with a neighboring pixel, and calculate loss only there.
    # Blind-spot leakage via receptive-field correlations is possible; this is a baseline only.
    mask=(torch.rand_like(x)<fraction)
    dy=random.choice([-1,0,1]); dx=random.choice([-1,0,1])
    if dx==0 and dy==0: dx=1
    replacement=torch.roll(x,shifts=(dy,dx),dims=(-2,-1))
    return torch.where(mask,replacement,x),mask

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/processed')
    p.add_argument('--out',default='results/baseline_smoke')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--patch',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    paths=sorted(Path(args.input).glob('slice_*.tif'))
    if len(paths)<4: raise RuntimeError('Need at least 4 TIFF slices')
    if args.patch<16: raise ValueError('patch must be >=16')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # Reserve last 4 contiguous slices for later held-out checks. Do not use them in training.
    train_paths=paths[:-4]; test_paths=paths[-4:]
    arrays=[]
    for path in train_paths:
        a=tifffile.imread(path)
        if a.ndim!=2 or min(a.shape)<args.patch: raise ValueError(f'Bad shape: {path} {a.shape}')
        arrays.append(a.astype(np.float32)/255.0)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=TinyCNN().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[]; start=time.perf_counter()
    print(f'Device: {device}; train slices: {len(train_paths)}; held-out slices: {len(test_paths)}',flush=True)
    for step in range(1,args.steps+1):
        a=random.choice(arrays); h,w=a.shape; y=random.randrange(h-args.patch+1); x=random.randrange(w-args.patch+1)
        crop=torch.from_numpy(a[y:y+args.patch,x:x+args.patch].copy())[None,None].to(device)
        corrupted,mask=mask_replace(crop)
        pred=model(corrupted)
        loss=((pred-crop).square()*mask).sum()/mask.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
        history.append((step,float(loss.detach().cpu())))
        if step==1 or step%10==0 or step==args.steps:
            print(f'Step {step}/{args.steps} masked-MSE={history[-1][1]:.6f}',flush=True)
    torch.save({'state_dict':model.cpu().state_dict(),'seed':args.seed,'width':32},out/'baseline.pt')
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','masked_mse']);writer.writerows(history)
    with (out/'run.json').open('w') as f:
        json.dump({'device':str(device),'steps':args.steps,'patch':args.patch,'seed':args.seed,
                   'train_files':[p.name for p in train_paths], 'heldout_files':[p.name for p in test_paths],
                   'seconds':round(time.perf_counter()-start,2),
                   'note':'Development smoke test; not publication validation. Downsampled input.'},f,indent=2)
    print(f'Done. Checkpoint: {out / "baseline.pt"}',flush=True)

if __name__=='__main__': main()
