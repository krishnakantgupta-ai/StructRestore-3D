#!/usr/bin/env python3
"""Full-resolution, memory-mapped FIB-SEM masked-pixel baseline training.
Development experiment, not a validated denoising benchmark.
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
        self.net=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.ReLU(),
                               nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
                               nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
                               nn.Conv2d(width,1,3,padding=1))
    def forward(self,x): return self.net(x)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/raw')
    p.add_argument('--out',default='results/baseline_fullres')
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--patch',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch',type=int,default=2)
    args=p.parse_args()
    if args.patch<16 or args.batch<1 or args.steps<1: p.error('patch>=16, batch>=1, steps>=1 required')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    paths=sorted(Path(args.input).glob('slice_*.tif'))
    if len(paths)<5: p.error('Need at least five TIFF slices')
    train_paths,heldout=paths[:-4],paths[-4:]
    # TIFF memmap reads cropped uint8 pixels from disk without loading entire images as float32.
    arrays=[]
    for path in train_paths:
        a=tifffile.memmap(path)
        if a.ndim!=2 or a.dtype!=np.uint8 or min(a.shape)<args.patch:
            raise ValueError(f'Expected 2D uint8 TIFF with dimensions >= patch: {path} {a.shape} {a.dtype}')
        arrays.append(a)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=TinyCNN().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    hist=[];start=time.perf_counter()
    print(f'Device: {device}; full-resolution images: {len(arrays)}; held out: {len(heldout)}; batch: {args.batch}',flush=True)
    for step in range(1,args.steps+1):
        patches=[]
        for _ in range(args.batch):
            a=random.choice(arrays);h,w=a.shape
            y=random.randrange(h-args.patch+1);x=random.randrange(w-args.patch+1)
            patches.append(np.asarray(a[y:y+args.patch,x:x+args.patch]).copy())
        clean=torch.from_numpy(np.stack(patches).astype(np.float32)/255.0)[:,None].to(device)
        mask=torch.rand_like(clean)<0.05
        # Per-sample independent spatial shift, always nonzero; mask only contributes to loss.
        replacement=torch.empty_like(clean)
        for b in range(args.batch):
            dy,dx=random.choice([(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0)])
            replacement[b]=torch.roll(clean[b],shifts=(dy,dx),dims=(-2,-1))
        corrupted=torch.where(mask,replacement,clean)
        pred=model(corrupted)
        loss=((pred-clean).square()*mask).sum()/mask.sum().clamp_min(1)
        opt.zero_grad();loss.backward();opt.step()
        v=float(loss.detach().cpu());hist.append((step,v))
        if step==1 or step%max(1,args.steps//10)==0 or step==args.steps:
            print(f'Step {step}/{args.steps} masked-MSE={v:.6f}',flush=True)
    torch.save({'state_dict':model.cpu().state_dict(),'seed':args.seed,'width':32},out/'baseline.pt')
    with (out/'history.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['step','masked_mse']);wr.writerows(hist)
    meta={'device':str(device),'steps':args.steps,'patch':args.patch,'batch':args.batch,
          'seed':args.seed,'train_files':[p.name for p in train_paths],
          'heldout_files':[p.name for p in heldout],
          'image_shape':list(arrays[0].shape),'seconds':round(time.perf_counter()-start,2),
          'note':'Full-resolution patch baseline. Held-out slices are adjacent and not independent biological replicates; training loss is not denoising validation.'}
    (out/'run.json').write_text(json.dumps(meta,indent=2))
    print('Done. Checkpoint:',out/'baseline.pt',flush=True)
if __name__=='__main__': main()
