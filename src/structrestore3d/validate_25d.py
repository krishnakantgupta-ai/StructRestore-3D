#!/usr/bin/env python3
"""Controlled added-Gaussian-noise validation for StructRestore-3D.
The reference is an ALREADY NOISY FIB-SEM slice, not clean ground truth.
Models are frozen. Three channels receive independently generated extra noise.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import tifffile
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_fullres import TinyCNN
from structrestore_25d import Net25D

def load_model(path, kind, device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if kind=='single':
        model=TinyCNN(width=int(ck['width']))
    else:
        if ck['mode']!=kind: raise ValueError(f'{path}: expected mode {kind}, found {ck["mode"]}')
        model=Net25D(mode=kind,width=int(ck['width']))
    model.load_state_dict(ck['state_dict'],strict=True)
    return model.to(device).eval()

def predict(model, input_array, device):
    # Crop includes a 16-pixel context margin; discard margin to avoid edge artifacts.
    with torch.inference_mode():
        inp=torch.from_numpy(input_array[None].copy()).to(device)
        return model(inp).clamp(0,1)[0,0].cpu().numpy()[16:-16,16:-16]

def metrics(ref, out):
    ref=np.asarray(ref,dtype=np.float32);out=np.asarray(out,dtype=np.float32)
    gy,gx=np.gradient(ref);py,px=np.gradient(out)
    edge_rmse=float(np.sqrt(np.mean((gx-px)**2+(gy-py)**2)))
    return dict(psnr=float(peak_signal_noise_ratio(ref,out,data_range=1)),
                ssim=float(structural_similarity(ref,out,data_range=1)),
                edge_rmse=edge_rmse)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/raw')
    p.add_argument('--single',default='results/baseline_fullres/baseline.pt')
    p.add_argument('--plain',default='results/25d_plain/model.pt')
    p.add_argument('--gated',default='results/25d_gated/model.pt')
    p.add_argument('--out',default='results/quant_validation')
    p.add_argument('--crop',type=int,default=512)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--sigmas',type=float,nargs='+',default=[0.05,0.10])
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if args.crop<64 or args.trials<1 or any(s<=0 or s>0.5 for s in args.sigmas):
        p.error('crop>=64, trials>=1 and 0<sigma<=0.5 required')
    files=sorted(Path(args.input).glob('slice_*.tif'))
    if len(files)!=16: raise ValueError('Expected the 16 original consecutive TIFF slices')
    for path in [args.single,args.plain,args.gated]:
        if not Path(path).is_file():raise FileNotFoundError(path)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    models={k:load_model(v,k,device) for k,v in [('single',args.single),('plain',args.plain),('gated',args.gated)]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    rows=[]; first_figure=False
    # Only targets whose both neighbors belong to held-out slice group.
    for z in (13,14):
        ims=[tifffile.memmap(str(files[k])) for k in (z-1,z,z+1)]
        h,w=ims[1].shape;side=args.crop;pad=16
        # Three spatially separated, deterministic regions; avoid image boundaries.
        centers=[(h//4,w//4),(h//2,w//2),(3*h//4,3*w//4)]
        for region,(cy,cx) in enumerate(centers):
            y=max(pad,min(h-side-pad,cy-side//2));x=max(pad,min(w-side-pad,cx-side//2))
            stack=np.stack([np.asarray(im[y-pad:y+side+pad,x-pad:x+side+pad],dtype=np.float32)/255 for im in ims])
            reference=stack[1,pad:-pad,pad:-pad]
            for sigma in args.sigmas:
                for trial in range(args.trials):
                    # Extra noise is independently sampled per section, unlike the natural FIB-SEM noise.
                    noisy=np.clip(stack+rng.normal(0,sigma,stack.shape).astype(np.float32),0,1)
                    candidates={'added_noise':noisy[1,pad:-pad,pad:-pad]}
                    timings={'added_noise':0.0}
                    for kind,model in models.items():
                        t0=time.perf_counter()
                        candidates[kind]=predict(model,noisy[1:2] if kind=='single' else noisy,device)
                        timings[kind]=time.perf_counter()-t0
                    for kind,pred in candidates.items():
                        m=metrics(reference,pred)
                        rows.append(dict(slice=f'slice_{z:04d}',region=region,y=y,x=x,
                            sigma=sigma,trial=trial,method=kind,seconds=round(timings[kind],5),**m))
                    if not first_figure:
                        fig,axes=plt.subplots(1,5,figsize=(15,3.5),constrained_layout=True)
                        for ax,(kind,img) in zip(axes,[('Reference*',reference),('Added noise',candidates['added_noise']),('Single',candidates['single']),('Plain 2.5D',candidates['plain']),('Gated 2.5D',candidates['gated'])]):
                            ax.imshow(img,cmap='gray',vmin=0,vmax=1);ax.set_title(kind);ax.axis('off')
                        fig.suptitle(f'slice_{z:04d}, sigma={sigma}, region={region} (*reference retains native noise)')
                        fig.savefig(out/'validation_example.png',dpi=160);plt.close(fig)
                        first_figure=True
        print(f'Finished slice_{z:04d}',flush=True)
    fields=['slice','region','y','x','sigma','trial','method','seconds','psnr','ssim','edge_rmse']
    with (out/'per_trial.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    summary=[]
    for sigma in args.sigmas:
        for method in ['added_noise','single','plain','gated']:
            subset=[r for r in rows if r['sigma']==sigma and r['method']==method]
            record={'sigma':sigma,'method':method,'n_patches':len(subset)}
            for key in ['psnr','ssim','edge_rmse','seconds']:
                vals=np.array([r[key] for r in subset],dtype=float)
                record[key+'_mean']=round(float(vals.mean()),6)
                record[key+'_sd']=round(float(vals.std(ddof=1)),6) if len(vals)>1 else 0
            summary.append(record)
    with (out/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=summary[0].keys());wr.writeheader();wr.writerows(summary)
    (out/'metadata.json').write_text(json.dumps(dict(device=str(device),seed=args.seed,crop=args.crop,
        trials=args.trials,sigmas=args.sigmas,heldout_targets=['slice_0013','slice_0014'],
        caveat='Reference is original noisy FIB-SEM. Scores measure recovery from ADDITIONAL simulated Gaussian noise only. Patches/trials from same specimen are correlated; not independent biological replicates. Models trained on native-noise images; this is an out-of-distribution stress test.'),indent=2))
    print('\nMEAN METRICS (higher PSNR/SSIM better; lower edge RMSE better)')
    for r in summary:
        print(f"sigma={r['sigma']:.2f} {r['method']:>11}: PSNR={r['psnr_mean']:.2f} SSIM={r['ssim_mean']:.4f} edge_RMSE={r['edge_rmse_mean']:.4f}")
    print('\nSaved:',out/'summary.csv',out/'per_trial.csv',out/'validation_example.png')

if __name__=='__main__':main()
