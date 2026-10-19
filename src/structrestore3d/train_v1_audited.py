#!/usr/bin/env python3
"""StructRestore-3D V1: auditable masked-input training, not a proven blind-spot denoiser.
Strict direct target-value exclusion at selected coordinates in ALL channels.
Neighbor correlations remain; never claim independent clean targets or guaranteed denoising.
Existing V0 checkpoints are never read or overwritten.
"""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, mode='gated', width=32):
        super().__init__()
        assert mode in ('single','plain','gated')
        self.mode=mode
        self.net=nn.Sequential(nn.Conv2d(1 if mode=='single' else 3,width,3,padding=1),nn.ReLU(),
          nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
          nn.Conv2d(width,width,3,padding=1),nn.ReLU(),nn.Conv2d(width,1,3,padding=1))
    def forward(self,x):
        if self.mode=='single': x=x[:,1:2]
        elif self.mode=='gated':
            c=x[:,1:2]; cm=F.avg_pool2d(c,5,stride=1,padding=2)
            p=x[:,0:1]; n=x[:,2:3]
            gp=torch.exp(-(F.avg_pool2d(p,5,1,2)-cm).abs()/0.12)
            gn=torch.exp(-(F.avg_pool2d(n,5,1,2)-cm).abs()/0.12)
            x=torch.cat((c+gp*(p-c),c,c+gn*(n-c)),1)
        return self.net(x)

def strict_mask(x, rate=0.05):
    """Independent filler and shared mask across channels, with no direct selected target value."""
    mask=torch.rand_like(x[:,1:2]) < rate
    filler=torch.rand_like(x)  # independent of target; same coordinates masked in all channels
    masked=torch.where(mask.expand_as(x),filler,x)
    return masked, x[:,1:2], mask

def audit(device='cpu'):
    torch.manual_seed(24)
    x=torch.rand(2,3,32,32,device=device,requires_grad=True)
    corrupted,target,mask=strict_mask(x,0.15)
    assert mask.any()
    assert torch.equal(corrupted[~mask.expand_as(x)],x[~mask.expand_as(x)])
    assert (corrupted[mask.expand_as(x)]-x[mask.expand_as(x)]).abs().min().item()>0
    for mode in ('single','plain','gated'):
        model=Model(mode,8).to(device)
        pred=model(corrupted)
        # Select one masked prediction, differentiate against original uncorrupted input.
        yx=torch.nonzero(mask[0,0])[0]; y=int(yx[0]); xx=int(yx[1])
        grad=torch.autograd.grad(pred[0,0,y,xx],x,retain_graph=True)[0]
        direct=grad[0,:,y,xx].abs().max().item()
        assert direct < 1e-10, f'Direct target gradient leakage in {mode}: {direct}'
        print(f'{mode}: direct target gradient={direct:.2g}; PASS')
    print('PASS: masked selected coordinates excluded from all input channels; direct gradients zero.')
    print('LIMITATION: adjacent pixels/slices remain correlated; this does NOT prove absence of indirect leakage.')

def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    files=sorted(Path(args.input).glob('slice_*.tif'))
    if len(files)<7:raise ValueError('Need at least 7 consecutive TIFFs')
    trainfiles=files[:-4]; heldout=files[-4:]
    imgs=[tifffile.memmap(str(f)) for f in trainfiles]
    for im in imgs:
        if im.ndim!=2 or im.dtype!=np.uint8 or min(im.shape)<args.patch:raise ValueError('Expected uint8 2D TIFF >= patch size')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Output folder not empty: {out}. Choose a NEW output directory; V0 must remain untouched.')
    out.mkdir(parents=True,exist_ok=True)
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=Model(args.mode,args.width).to(dev);opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    hist=[];t0=time.perf_counter(); rejected=0
    print(f'Mode={args.mode}, device={dev}, train={len(trainfiles)}, held-out={len(heldout)}',flush=True)
    for step in range(1,args.steps+1):
        batch=[];stds=[]
        for b in range(args.batch):
            for attempt in range(args.max_attempts):
                z=random.randrange(1,len(imgs)-1)
                h,w=imgs[z].shape
                y=random.randrange(h-args.patch+1);xx=random.randrange(w-args.patch+1)
                center=np.asarray(imgs[z][y:y+args.patch,xx:xx+args.patch],dtype=np.float32)/255
                std=float(center.std())
                if std>=args.min_std:break
                rejected+=1
            else:
                raise RuntimeError(f'Could not find informative patch after {args.max_attempts} tries. Reduce --min-std.')
            batch.append(np.stack([np.asarray(imgs[j][y:y+args.patch,xx:xx+args.patch],dtype=np.float32)/255 for j in (z-1,z,z+1)]))
            stds.append(std)
        x=torch.from_numpy(np.stack(batch)).to(dev)
        corrupted,target,mask=strict_mask(x,args.mask_rate)
        pred=model(corrupted)
        loss=((pred-target).square()*mask).sum()/mask.sum().clamp_min(1)
        opt.zero_grad();loss.backward();opt.step()
        row={'step':step,'masked_mse':float(loss.detach().cpu()),'patch_std_mean':float(np.mean(stds)),
             'patch_std_min':float(min(stds)),'masked_pixels':int(mask.sum().item()),
             'masked_fraction':float(mask.float().mean().item()),'rejected_patches_cumulative':rejected}
        hist.append(row)
        if step==1 or step%50==0 or step==args.steps:
            print(f'Step {step}/{args.steps} masked-MSE={row["masked_mse"]:.6f} patch-std={row["patch_std_mean"]:.4f} rejected={rejected}',flush=True)
    torch.save({'state_dict':model.cpu().state_dict(),'mode':args.mode,'width':args.width,
                'seed':args.seed,'version':'v1_strict_direct_mask'},out/'model.pt')
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=hist[0].keys());writer.writeheader();writer.writerows(hist)
    report={'version':'v1_strict_direct_mask','mode':args.mode,'device':str(dev),'steps':args.steps,'patch':args.patch,
            'batch':args.batch,'seed':args.seed,'mask_rate':args.mask_rate,'min_std':args.min_std,
            'train_files':[p.name for p in trainfiles],'heldout_files':[p.name for p in heldout],
            'total_rejected_patches':rejected,'seconds':round(time.perf_counter()-t0,2),
            'limitations':['Direct masked target excluded from all three input channels, not guaranteed blind-spot independence.',
                           'Adjacent section information remains correlated.',
                           'Inference on unmasked images has train/test corruption mismatch.',
                           'Existing adjacent held-out slices are not independent biological replicates.']}
    (out/'run.json').write_text(json.dumps(report,indent=2))
    print('Saved:',out/'model.pt',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['audit','train'])
    ap.add_argument('--input',default='data/raw');ap.add_argument('--out',default='results/v1_gated')
    ap.add_argument('--mode',choices=['single','plain','gated'],default='gated')
    ap.add_argument('--steps',type=int,default=500);ap.add_argument('--patch',type=int,default=128)
    ap.add_argument('--batch',type=int,default=2);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--width',type=int,default=32);ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--mask-rate',type=float,default=0.05);ap.add_argument('--min-std',type=float,default=0.04)
    ap.add_argument('--max-attempts',type=int,default=200)
    a=ap.parse_args()
    if a.command=='audit':audit()
    else:
        if not 0<a.mask_rate<1:raise ValueError('mask rate must be in (0,1)')
        train(a)
if __name__=='__main__':main()
