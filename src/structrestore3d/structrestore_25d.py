#!/usr/bin/env python3
"""StructRestore-3D experimental 2.5D ablation. Requires torch, numpy, tifffile, matplotlib.
Train on 3 consecutive raw uint8 TIFFs, target only masked central pixels.
The reliability gate is a fixed local intensity-agreement heuristic (not learned).
"""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch import nn
import torch.nn.functional as F

class Net25D(nn.Module):
    def __init__(self, mode='plain', width=32):
        super().__init__()
        if mode not in ('plain','gated'): raise ValueError(mode)
        self.mode=mode
        self.net=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.ReLU(),
             nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
             nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
             nn.Conv2d(width,1,3,padding=1))
    def forward(self, x):
        if self.mode=='gated':
            # Neighbor trust estimated from low-pass agreement; no clean target needed.
            c=x[:,1:2]
            cmean=F.avg_pool2d(c,5,stride=1,padding=2)
            prev=x[:,0:1]; nxt=x[:,2:3]
            gp=torch.exp(-torch.abs(F.avg_pool2d(prev,5,1,2)-cmean)/0.12)
            gn=torch.exp(-torch.abs(F.avg_pool2d(nxt,5,1,2)-cmean)/0.12)
            x=torch.cat((c+gp*(prev-c),c,c+gn*(nxt-c)),dim=1)
        return self.net(x)

def device_choice():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def corrupt_center(x, rate=0.05):
    target=x[:,1:2].clone()
    mask=torch.rand_like(target)<rate
    dy,dx=random.choice([(a,b) for a in (-1,0,1) for b in (-1,0,1) if a or b])
    shifted=torch.roll(target,shifts=(dy,dx),dims=(-2,-1))
    corrupted=x.clone()
    corrupted[:,1:2]=torch.where(mask,shifted,target)
    return corrupted,target,mask

def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    files=sorted(Path(args.input).glob('slice_*.tif'))
    if len(files)<7: raise ValueError('At least 7 sequential TIFFs required')
    trainfiles=files[:-4]; testfiles=files[-4:]
    if len(trainfiles)<3: raise ValueError('Need 3 training slices')
    # Memmap uncompressed TIFFs to avoid materializing whole stack as float32.
    imgs=[tifffile.memmap(str(f)) for f in trainfiles]
    for im in imgs:
        if im.ndim!=2 or im.dtype!=np.uint8 or min(im.shape)<args.patch: raise ValueError('Expected uint8 2D TIFF >= patch size')
    dev=device_choice(); net=Net25D(args.mode,args.width).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    hist=[]; t0=time.perf_counter()
    print(f'Mode: {args.mode}; device: {dev}; train: {len(trainfiles)} slices; held out: {len(testfiles)}',flush=True)
    for step in range(1,args.steps+1):
        batch=[]
        for _ in range(args.batch):
            z=random.randrange(1,len(imgs)-1)
            h,w=imgs[z].shape
            y=random.randrange(h-args.patch+1);x=random.randrange(w-args.patch+1)
            batch.append(np.stack([np.asarray(imgs[k][y:y+args.patch,x:x+args.patch],dtype=np.float32)/255.0 for k in (z-1,z,z+1)]))
        x=torch.from_numpy(np.stack(batch)).to(dev)
        corrupted,target,mask=corrupt_center(x)
        pred=net(corrupted)
        loss=((pred-target).square()*mask).sum()/mask.sum().clamp_min(1)
        opt.zero_grad();loss.backward();opt.step()
        hist.append((step,float(loss.detach().cpu())))
        if step==1 or step%50==0 or step==args.steps: print(f'Step {step}/{args.steps} masked-MSE={hist[-1][1]:.6f}',flush=True)
    torch.save({'state_dict':net.cpu().state_dict(),'width':args.width,'mode':args.mode,'seed':args.seed},out/'model.pt')
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','masked_mse']);writer.writerows(hist)
    with (out/'run.json').open('w') as f:
        json.dump({'mode':args.mode,'device':str(dev),'steps':args.steps,'patch':args.patch,'batch':args.batch,'seed':args.seed,
                   'train_files':[p.name for p in trainfiles], 'heldout_files':[p.name for p in testfiles],
                   'seconds':round(time.perf_counter()-t0,2),'parameters':sum(p.numel() for p in net.parameters()),
                   'note':'Prototype ablation; neighboring slices are correlated, not independent validation.'},f,indent=2)
    print('Saved:',out/'model.pt',flush=True)

def infer(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=sorted(Path(args.input).glob('slice_*.tif'))
    if len(files)<7:raise ValueError('Need >=7 slices')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=Net25D(ck['mode'],ck['width']);net.load_state_dict(ck['state_dict'])
    dev=device_choice();net=net.to(dev).eval()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Last four slices are held out. Only internal slices with held-out neighbors are evaluated.
    # This excludes first held-out slice to avoid using training slice as its neighbor.
    # Last slice excluded because next neighbor unavailable.
    targets=[len(files)-3,len(files)-2]
    results=[]
    for z in targets:
        ims=[tifffile.memmap(str(files[k])) for k in (z-1,z,z+1)]
        h,w=ims[1].shape;prediction=np.empty((h,w),dtype=np.uint8)
        t0=time.perf_counter()
        # Inference on overlapping tiles with a 16px halo; retain central tile only.
        tile=args.tile;halo=16
        with torch.inference_mode():
            for y in range(0,h,tile):
                for x in range(0,w,tile):
                    y0=max(0,y-halo);x0=max(0,x-halo)
                    y1=min(h,y+tile+halo);x1=min(w,x+tile+halo)
                    crop=np.stack([np.asarray(im[y0:y1,x0:x1],dtype=np.float32)/255 for im in ims])
                    tensor=torch.from_numpy(crop[None]).to(dev)
                    den=net(tensor).clamp(0,1)[0,0].cpu().numpy()
                    ey=min(y+tile,h);ex=min(x+tile,w)
                    prediction[y:ey,x:ex]=np.rint(255*den[y-y0:ey-y0,x-x0:ex-x0]).astype(np.uint8)
        name=files[z].stem
        tifffile.imwrite(out/f'{name}_denoised.tif',prediction)
        orig=np.asarray(ims[1]);mae=float(np.abs(prediction.astype(np.float32)-orig.astype(np.float32)).mean()/255)
        # Cropped visual figure: avoid misleading shrink-to-fit full-res comparison.
        cy,cx=h//2,w//2;s=args.crop
        yy=max(0,min(h-s,cy-s//2));xx=max(0,min(w-s,cx-s//2))
        a=orig[yy:yy+s,xx:xx+s];b=prediction[yy:yy+s,xx:xx+s]
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(a,cmap='gray',vmin=0,vmax=255);axes[0].set_title('Original (full resolution)')
        axes[1].imshow(b,cmap='gray',vmin=0,vmax=255);axes[1].set_title(f'2.5D {ck["mode"]}')
        im=axes[2].imshow((b.astype(float)-a.astype(float))/255,cmap='coolwarm',vmin=-0.4,vmax=0.4)
        axes[2].set_title('Output minus input');fig.colorbar(im,ax=axes[2],fraction=.045)
        for ax in axes:ax.axis('off')
        fig.suptitle(f'{name}: full-resolution {s}x{s} crop');fig.tight_layout()
        fig.savefig(out/f'{name}_comparison.png',dpi=150);plt.close(fig)
        seconds=time.perf_counter()-t0;results.append((name,round(seconds,2),round(mae,6)))
        print(f'{name}: {seconds:.2f}s, mean absolute change={mae:.5f}',flush=True)
    with (out/'diagnostics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['slice','seconds','mean_absolute_change']);writer.writerows(results)
    print('Saved to',out)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    t=sub.add_parser('train');t.add_argument('--input',default='data/raw');t.add_argument('--out',required=True)
    t.add_argument('--mode',choices=['plain','gated'],required=True)
    t.add_argument('--steps',type=int,default=500);t.add_argument('--patch',type=int,default=128)
    t.add_argument('--batch',type=int,default=2);t.add_argument('--width',type=int,default=32)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--lr',type=float,default=.001)
    v=sub.add_parser('infer');v.add_argument('--input',default='data/raw');v.add_argument('--checkpoint',required=True)
    v.add_argument('--out',required=True);v.add_argument('--tile',type=int,default=384);v.add_argument('--crop',type=int,default=512)
    args=p.parse_args()
    if args.action=='train':train(args)
    else:infer(args)
if __name__=='__main__':main()
