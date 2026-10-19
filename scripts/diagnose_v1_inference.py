#!/usr/bin/env python3
"""StructRestore-3D Step 28B: read-only diagnosis of V1 inference.

Uses existing V1 checkpoints. No training. No original files modified.
The native FIB-SEM slice is noisy, NOT verified clean ground truth.

Diagnoses:
  - Unmasked vs correctly masked-partition inference on the SAME V1 checkpoint.
  - Sensitivity to random mask/filler realizations (fixed input image).
  - Neighboring-slice alignment as a correlation diagnostic (not proof of misregistration).
  - Direct target leakage in the trained V1 checkpoint, at sampled target coordinates.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name != 'scripts' and (SCRIPT_DIR/'scripts').is_dir():
    SCRIPT_DIR = SCRIPT_DIR/'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from train_v1_audited import Model, strict_mask

CKPTS = {
    'single': 'results/v1_single/model.pt',
    'plain': 'results/v1_plain/model.pt',
    'gated': 'results/v1_gated/model.pt',
}


def scores(ref, img):
    ref = ref.astype(np.float32, copy=False)
    img = np.clip(img, 0, 1).astype(np.float32, copy=False)
    gy, gx = np.gradient(ref)
    py, px = np.gradient(img)
    return dict(psnr=float(peak_signal_noise_ratio(ref, img, data_range=1)),
                ssim=float(structural_similarity(ref, img, data_range=1)),
                edge_rmse=float(np.sqrt(np.mean((gy-py)**2 + (gx-px)**2))))


def correlation(a, b):
    aa = (a.astype(np.float64) - np.mean(a)).ravel()
    bb = (b.astype(np.float64) - np.mean(b)).ravel()
    den = np.linalg.norm(aa)*np.linalg.norm(bb)
    return float(np.dot(aa, bb)/den) if den > 1e-10 else None


def translated_overlap(a, b, dy, dx):
    # Compare a[y,x] to b[y+dy,x+dx], without wrapping pixels.
    h, w = a.shape
    y0, y1 = max(0, -dy), min(h, h-dy)
    x0, x1 = max(0, -dx), min(w, w-dx)
    return a[y0:y1,x0:x1], b[y0+dy:y1+dy,x0+dx:x1+dx]


def neighbor_alignment(stack, radius=3):
    # Low-pass correlations are diagnostic; anatomical changes can also change correlation.
    smooth = [gaussian_filter(s, 1.5) for s in stack]
    result = {}
    for name, neighbor in [('previous', 0), ('next', 2)]:
        center, other = smooth[1], smooth[neighbor]
        direct = correlation(center, other)
        best = (-np.inf, 0, 0)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                a,b = translated_overlap(center, other, dy, dx)
                c = correlation(a,b)
                if c is not None and c > best[0]:best=(c,dy,dx)
        result[name] = dict(unshifted_correlation=direct,
                            best_correlation=float(best[0]),
                            best_relative_shift_pixels=dict(dy=int(best[1]),dx=int(best[2])))
    return result


def infer_unmasked(model, stack_tensor, pad):
    with torch.inference_mode():
        pred=model(stack_tensor).clamp(0,1)[0,0].cpu().numpy()
    return pred[pad:-pad,pad:-pad].copy()


def infer_masked(model, stack_tensor, pad, groups, ids, seed):
    _, _, h, w = stack_tensor.shape
    out=np.empty((h-2*pad,w-2*pad),dtype=np.float32)
    assigned=np.zeros_like(out,dtype=np.uint8)
    # Seeding at each call ensures all modes get exactly the same masks and fillers.
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        for g in range(groups):
            selected=(ids==g)
            inside=selected[pad:-pad,pad:-pad]
            if not inside.any():continue
            mask=torch.from_numpy(selected[None,None]).to(stack_tensor.device)
            filler=torch.rand_like(stack_tensor)
            corrupted=torch.where(mask.expand_as(stack_tensor),filler,stack_tensor)
            pred=model(corrupted).clamp(0,1)[0,0].cpu().numpy()[pad:-pad,pad:-pad]
            out[inside]=pred[inside]
            assigned[inside]+=1
    if not np.all(assigned==1):
        raise RuntimeError('Partition failure: every output pixel must be predicted once')
    return out


def check_trained_direct_leakage(models,dev):
    torch.manual_seed(247)
    x=torch.rand((1,3,40,40),device=dev,requires_grad=True)
    corrupted,_,mask=strict_mask(x,rate=0.1)
    coords=torch.nonzero(mask[0,0])
    if len(coords)==0:raise RuntimeError('Mask test selected no pixels')
    coords=coords[:5]
    checks={}
    for mode,model in models.items():
        max_g=0.
        for i, coord in enumerate(coords):
            y, xx = [int(v) for v in coord]
            pred=model(corrupted)
            grads=torch.autograd.grad(pred[0,0,y,xx],x,retain_graph=True)[0]
            val=float(grads[0,:,y,xx].abs().max().detach().cpu())
            max_g=max(max_g,val)
        checks[mode]={'max_direct_target_gradient':max_g,'passed':bool(max_g<1e-10)}
        print(f'CHECK trained {mode}: max direct target gradient={max_g:.3g};',
              'PASS' if checks[mode]['passed'] else 'FAIL', flush=True)
    if not all(x['passed'] for x in checks.values()):
        raise RuntimeError('Direct-masking audit failed')
    return checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='data/raw')
    ap.add_argument('--out',default='results/v1_inference_diagnostic')
    ap.add_argument('--models-root',default='.')
    ap.add_argument('--slice-index',type=int,default=13,choices=[13,14])
    ap.add_argument('--region',type=int,choices=[0,1,2],default=0,help='Match Step 27 region 0/1/2; default 0 reproduces comparison figure')
    ap.add_argument('--crop',type=int,default=256)
    ap.add_argument('--groups',type=int,default=20)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--sigma',type=float,default=0.05,help='Additional Gaussian noise on all three sections')
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if args.crop<64 or not 2<=args.groups<=255 or not 2<=args.repeats<=10 or not 0<args.sigma<0.5:
        ap.error('Require crop>=64, 2<=groups<=255, 2<=repeats<=10, 0<sigma<0.5')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty. Choose a new --out directory.')
    filenames=sorted(Path(args.input).glob('slice_*.tif'))
    if len(filenames)!=16:raise ValueError(f'Expected 16 original TIFFs; found {len(filenames)}')
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpaths={m:Path(args.models_root)/p for m,p in CKPTS.items()}
    missing=[str(p) for p in ckpaths.values() if not p.exists()]
    if missing:raise FileNotFoundError('Missing checkpoints: '+', '.join(missing))
    models={}
    for mode,path in ckpaths.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('mode')!=mode:raise ValueError(f'Wrong mode in {path}')
        model=Model(mode,int(checkpoint.get('width',32)))
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        models[mode]=model.to(dev).eval()
        print('Loaded:',mode,path,flush=True)
    checks=check_trained_direct_leakage(models,dev)
    pad=16
    z=args.slice_index
    ims=[tifffile.memmap(str(filenames[j])) for j in (z-1,z,z+1)]
    H,W=ims[1].shape
    if min(H,W)<args.crop+2*pad:raise ValueError('Crop too large for TIFF')
    centers=[(H//4,W//4),(H//2,W//2),(3*H//4,3*W//4)]
    cy,cx=centers[args.region]
    y=max(pad,min(H-args.crop-pad,cy-args.crop//2))
    x=max(pad,min(W-args.crop-pad,cx-args.crop//2))
    # Read only selected crops, rather than entire full-resolution images into RAM.
    stack=np.stack([np.asarray(im[y-pad:y+args.crop+pad,x-pad:x+args.crop+pad],dtype=np.float32)/255. for im in ims])
    ref=stack[1,pad:-pad,pad:-pad].copy()
    alignment=neighbor_alignment(stack[:,pad:-pad,pad:-pad])
    print('Neighboring-section correlations:',json.dumps(alignment),flush=True)
    rng=np.random.default_rng(args.seed)
    noisy=np.clip(stack+rng.normal(0,args.sigma,size=stack.shape).astype(np.float32),0,1)
    # Both inference modes operate on exactly the same simulated noisy input.
    inp=torch.from_numpy(np.ascontiguousarray(noisy[None])).to(dev)
    raw=noisy[1,pad:-pad,pad:-pad]
    predictions={'added_noise':raw,'gaussian_0p8':gaussian_filter(raw,0.8)}
    rows=[]; variations=[];first_masked={}
    for name,img in predictions.items():
        rows.append(dict(mode=name,inference='baseline',repeat=0,**scores(ref,img)))
    for mode,model in models.items():
        unmasked=infer_unmasked(model,inp,pad)
        predictions[mode+'_unmasked']=unmasked
        rows.append(dict(mode=mode,inference='unmasked',repeat=0,**scores(ref,unmasked)))
        pred_runs=[]
        for rep in range(args.repeats):
            # Sample a fresh disjoint mask partition; fixed noisy input for all repeats.
            ids=np.random.default_rng(args.seed+1009*rep).integers(0,args.groups,
                    size=noisy.shape[1:],dtype=np.uint8)
            pred=infer_masked(model,inp,pad,args.groups,ids,args.seed+3001*rep)
            rows.append(dict(mode=mode,inference='masked',repeat=rep,**scores(ref,pred)))
            pred_runs.append(pred)
        predictions[mode+'_masked']=pred_runs[0]
        first_masked[mode]=pred_runs[0]
        diffs=[np.abs(pr-pred_runs[0]) for pr in pred_runs[1:]]
        differences=np.stack(diffs)
        variations.append(dict(mode=mode,
                mean_abs_prediction_difference=float(differences.mean()),
                p95_abs_prediction_difference=float(np.percentile(differences,95)),
                max_abs_prediction_difference=float(differences.max())))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'per_repeat.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summaries=[]
    for name in ['added_noise','gaussian_0p8']+list(models):
        modes=['baseline'] if name in ('added_noise','gaussian_0p8') else ['unmasked','masked']
        for imode in modes:
            subset=[row for row in rows if row['mode']==name and row['inference']==imode]
            summaries.append(dict(method=name,inference=imode,n=len(subset),
                **{k+'_mean':float(np.mean([row[k] for row in subset])) for k in ('psnr','ssim','edge_rmse')}))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0].keys());writer.writeheader();writer.writerows(summaries)
    with (out/'mask_variability.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=variations[0].keys());writer.writeheader();writer.writerows(variations)
    figure=[('Native noisy reference',ref),('Added noise',raw),('Gaussian 0.8',predictions['gaussian_0p8'])]
    for mode in ('single','plain','gated'):
        figure.extend([(mode+' unmasked',predictions[mode+'_unmasked']),
                       (mode+' masked',predictions[mode+'_masked'])])
    fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
    for ax,(name,img) in zip(axes.ravel(),figure):
        ax.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(name);ax.axis('off')
    fig.suptitle(f'Inference-only diagnosis | slice_{z:04d} | sigma={args.sigma} | native reference NOT clean')
    fig.savefig(out/'inference_comparison.png',dpi=140)
    plt.close(fig)
    report=dict(experiment='V1 same-checkpoint inference comparison',source_slice=z,region=args.region,top_left_yx=[y,x],
      reference='Native noisy FIB-SEM TIFF, not clean truth',extra_gaussian_sigma=args.sigma,
      groups=args.groups,repeats=args.repeats,seed=args.seed,alignment=alignment,
      trained_checkpoint_direct_mask_tests=checks,
      checkpoints={k:str(v) for k,v in ckpaths.items()},
      limitations=[
          'The masked inference method is closer to the V1 training corruption, while unmasked inference is distribution-shifted.',
          'Masked and unmasked results test inference protocol sensitivity, not comparative denoising efficacy.',
          'The original noisy reference is NOT clean; PSNR, SSIM and edge-RMSE are fidelity measures only.',
          'Even if the direct gradient test passes, adjacent pixels/slices remain correlated.',
          'Intensity cross-correlation of adjacent sections does not distinguish misregistration from genuine structural changes.',
          'All evaluation regions are from one physical stack; not independent biological replicates.'
      ])
    (out/'metadata.json').write_text(json.dumps(report,indent=2))
    print('\nSame V1 checkpoint; higher PSNR/SSIM and lower edge RMSE = more faithful to NATIVE noisy reference:')
    for r in summaries:
        print(f"  {r['method']:<14} {r['inference']:<9} PSNR={r['psnr_mean']:6.2f} "
              f"SSIM={r['ssim_mean']:.4f} edge_RMSE={r['edge_rmse_mean']:.4f}")
    print('Mask-partition sensitivity (mean absolute difference between repeated runs):')
    for r in variations:print(f"  {r['mode']}: {r['mean_abs_prediction_difference']:.6f}")
    print('Saved:',out/'summary.csv',out/'mask_variability.csv',out/'inference_comparison.png',flush=True)

if __name__=='__main__':main()
