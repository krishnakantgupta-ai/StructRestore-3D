#!/usr/bin/env python3
"""StructRestore-3D Step 29: inference-only integer-translation registration probe.

No retraining. Uses 4 saved 2.5D checkpoints and held-out 3-slice stacks.
Registration is estimated from the SAME synthetically corrupted input (not from
clean ground truth), without wraparound or interpolation smoothing. The native
slice remains noisy: metrics quantify closeness to it, not denoising truth.

Both aligned/unaligned V1 passes use IDENTICAL pixel-group masks and fillers.
V0 uses original unmasked inference, V1 uses masked partition inference.
Alignment at inference only differs from training, so poor results don't rule
out the value of training on properly registered stacks.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import tifffile, torch
from scipy.ndimage import gaussian_filter, shift as nd_shift
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTS=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPTS))
from structrestore_25d import Net25D
from train_v1_audited import Model

CHECKPOINTS={
    'v0_plain': 'results/25d_plain/model.pt',
    'v0_gated': 'results/25d_gated/model.pt',
    'v1_plain': 'results/v1_plain/model.pt',
    'v1_gated': 'results/v1_gated/model.pt',
}

def corr(x,y):
    a=x.astype(np.float64).ravel();b=y.astype(np.float64).ravel()
    a-=a.mean();b-=b.mean()
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.dot(b)/norm) if norm>1e-12 else None

def overlapping(a,b,dy,dx):
    # Estimate shift (dy,dx): aligned moving[y,x] == moving[y+dy,x+dx].
    h,w=a.shape
    y0=max(0,-dy);y1=min(h,h-dy)
    x0=max(0,-dx);x1=min(w,w-dx)
    return a[y0:y1,x0:x1],b[y0+dy:y1+dy,x0+dx:x1+dx]

def register_stack(noisy,maxshift=6,blur=1.5):
    # Estimate on noisy inputs, not on the native-noise reference.
    smooth=[gaussian_filter(s,blur) for s in noisy]
    new=noisy.copy()
    details={}
    for idx,which in ((0,'previous'),(2,'next')):
        base=smooth[1]; other=smooth[idx]
        # Fair correlations on same central support to avoid overlap-size bias.
        margin=maxshift
        if min(base.shape)<=2*margin+16:raise ValueError('Crop too small for shift radius')
        central=base[margin:-margin,margin:-margin]
        best=(-np.inf,0,0)
        for dy in range(-maxshift,maxshift+1):
            for dx in range(-maxshift,maxshift+1):
                shifted=other[margin+dy:other.shape[0]-margin+dy,
                              margin+dx:other.shape[1]-margin+dx]
                val=corr(central,shifted)
                if val is not None and val>best[0]:best=(val,dy,dx)
        direct=corr(central,other[margin:-margin,margin:-margin])
        val,dy,dx=best
        # Exact integer-pixel translation, no interpolation / wrap-around.
        new[idx]=nd_shift(noisy[idx],shift=(-dy,-dx),order=0,mode='nearest',prefilter=False)
        details[which]={'dy':int(dy),'dx':int(dx),
                        'correlation_unregistered':float(direct),
                        'correlation_registered':float(val),
                        'correlation_gain':float(val-direct),
                        'shift_at_search_boundary':bool(abs(dy)==maxshift or abs(dx)==maxshift)}
    return new,details

def calc(ref,pred):
    pred=np.clip(pred,0,1).astype(np.float32)
    gy,gx=np.gradient(ref);py,px=np.gradient(pred)
    return {'psnr':float(peak_signal_noise_ratio(ref,pred,data_range=1.0)),
            'ssim':float(structural_similarity(ref,pred,data_range=1.0)),
            'edge_rmse':float(np.sqrt(np.mean((gy-py)**2+(gx-px)**2)))}

def load_models(root,dev):
    out={}
    for key,relative in CHECKPOINTS.items():
        ckpath=Path(root)/relative
        ck=torch.load(str(ckpath),map_location='cpu',weights_only=True)
        arch=key.split('_')[1]
        if ck.get('mode')!=arch:
            raise ValueError(f'{ckpath} contains mode {ck.get("mode")}, expected {arch}')
        net=Net25D(arch,int(ck.get('width',32))) if key.startswith('v0') else Model(arch,int(ck.get('width',32)))
        net.load_state_dict(ck['state_dict'],strict=True)
        out[key]=net.to(dev).eval()
        print('Loaded:',key,'<-',ckpath,flush=True)
    return out

def infer_v0(net,stack,dev,pad):
    with torch.inference_mode():
        x=torch.from_numpy(np.ascontiguousarray(stack[None])).to(dev)
        out=net(x).clamp(0,1)[0,0,pad:-pad,pad:-pad].cpu().numpy()
    return out.copy()

def infer_v1(net,stack,dev,pad,mask_ids,groups,fillers):
    _,h,w=stack.shape
    pred=np.empty((h-2*pad,w-2*pad),dtype=np.float32)
    coverage=np.zeros_like(pred,dtype=np.uint8)
    base=torch.from_numpy(np.ascontiguousarray(stack[None])).to(dev)
    with torch.inference_mode():
        for g in range(groups):
            pixels=(mask_ids==g)
            inside=pixels[pad:-pad,pad:-pad]
            if not inside.any():continue
            mask=torch.from_numpy(pixels[None,None]).to(dev)
            filler=torch.from_numpy(fillers[g]).to(dev)
            corrupted=torch.where(mask.expand_as(base),filler,base)
            result=net(corrupted).clamp(0,1)[0,0,pad:-pad,pad:-pad].cpu().numpy()
            pred[inside]=result[inside]
            coverage[inside]+=1
    if not np.all(coverage==1):raise RuntimeError('Mask partitions do not cover every output pixel once')
    return pred

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='data/raw')
    ap.add_argument('--out',default='results/step29_registration')
    ap.add_argument('--models-root',default='.')
    ap.add_argument('--crop',type=int,default=256)
    ap.add_argument('--shift-radius',type=int,default=6)
    ap.add_argument('--groups',type=int,default=20)
    ap.add_argument('--sigmas',type=float,nargs='+',default=[0.05])
    ap.add_argument('--regions',type=int,default=1,choices=[1,3],help='1 quick test or 3 regions per held-out slice')
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if args.crop<64 or not 1<=args.shift_radius<=12 or not 2<=args.groups<=255 or any(not 0<s<0.5 for s in args.sigmas):
        ap.error('Require crop >=64, shift radius 1..12, groups 2..255, sigma 0..0.5')
    if args.crop<=2*args.shift_radius+20:ap.error('Crop too small relative to shift radius')
    output=Path(args.out)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Output is not empty: {output}; select another --out')
    input_files=sorted(Path(args.input).glob('slice_*.tif'))
    if len(input_files)!=16:raise ValueError(f'Expected 16 TIFF slices, got {len(input_files)}')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:',device,flush=True)
    models=load_models(args.models_root,device)
    rng=np.random.default_rng(args.seed)
    results=[];align_rows=[];figure=None
    # Both neighbors always within the held-out partition 0012-0015.
    for z in (13,14):
        slices=[tifffile.memmap(str(input_files[k])) for k in (z-1,z,z+1)]
        H,W=slices[1].shape
        pad=16+args.shift_radius
        centers=[(H//4,W//4),(H//2,W//2),(3*H//4,3*W//4)][:args.regions]
        for region,(cy,cx) in enumerate(centers):
            top=max(pad,min(H-args.crop-pad,cy-args.crop//2))
            left=max(pad,min(W-args.crop-pad,cx-args.crop//2))
            stack=np.stack([np.asarray(im[top-pad:top+args.crop+pad,left-pad:left+args.crop+pad],dtype=np.float32)/255
                            for im in slices])
            ref=stack[1,pad:-pad,pad:-pad].copy()
            for sigma in args.sigmas:
                noisy=np.clip(stack+rng.normal(0,sigma,stack.shape).astype(np.float32),0,1)
                registered,detail=register_stack(noisy,args.shift_radius)
                align_rows.append({'slice':z,'region':region,'sigma':sigma,
                                   **{f'{kind}_{k}':v for kind,dictn in detail.items() for k,v in dictn.items()}})
                for kind,data in (('noisy_input',noisy),('registered_input_center',registered)):
                    if kind=='registered_input_center':
                        # The central slice must never be modified during neighbor registration.
                        assert np.array_equal(data[1],noisy[1])
                        continue
                    val=calc(ref,data[1,pad:-pad,pad:-pad])
                    results.append({'slice':z,'region':region,'sigma':sigma,'model':'noisy_input','registration':'none',**val})
                # Exactly shared mask IDs and per-group replacement tensors in every
                # masked model and alignment condition. 4 x (groups x H x W x 3)
                # would waste RAM, so sample on demand per mask group, cache only
                # the small groups needed for this single crop.
                h,w=noisy.shape[1:]
                mask_ids=rng.integers(0,args.groups,size=(h,w),dtype=np.uint8)
                fillers=rng.random((args.groups,1,3,h,w),dtype=np.float32)
                pred_for_figure={}
                for name,model in models.items():
                    for flag,data in (('unaligned',noisy),('registered',registered)):
                        if name.startswith('v0'):
                            pred=infer_v0(model,data,device,pad)
                        else:
                            pred=infer_v1(model,data,device,pad,mask_ids,args.groups,fillers)
                        result=calc(ref,pred)
                        results.append({'slice':z,'region':region,'sigma':sigma,'model':name,'registration':flag,**result})
                        if figure is None:
                            pred_for_figure[name+'_'+flag]=pred
                if figure is None:
                    fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
                    chart=[('Native reference (not clean)',ref),('Added Gaussian noise',noisy[1,pad:-pad,pad:-pad]),
                           ('V0 plain: unaligned',pred_for_figure['v0_plain_unaligned']),
                           ('V0 plain: registered',pred_for_figure['v0_plain_registered']),
                           ('V0 gated: unaligned',pred_for_figure['v0_gated_unaligned']),
                           ('V0 gated: registered',pred_for_figure['v0_gated_registered']),
                           ('V1 plain: unaligned',pred_for_figure['v1_plain_unaligned']),
                           ('V1 plain: registered',pred_for_figure['v1_plain_registered']),
                           ('V1 gated: registered',pred_for_figure['v1_gated_registered'])]
                    for ax,(label,img) in zip(axes.ravel(),chart):
                        ax.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(label,fontsize=11);ax.axis('off')
                    fig.suptitle(f'Step 29 | slice_{z:04d} region {region} sigma={sigma} | registered neighbors only')
                    figure=fig
                print(f'Slice {z} region {region} sigma={sigma:.2f}: shifts prev={detail["previous"]["dy"],detail["previous"]["dx"]} '
                      f'next={detail["next"]["dy"],detail["next"]["dx"]}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per_region.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
    with (output/'alignment.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(align_rows[0]));wr.writeheader();wr.writerows(align_rows)
    summary=[]
    for sigma in args.sigmas:
        for key in ('noisy_input',)+tuple(models):
            for reg in (('none',) if key=='noisy_input' else ('unaligned','registered')):
                subset=[r for r in results if r['sigma']==sigma and r['model']==key and r['registration']==reg]
                if not subset:continue
                row={'sigma':sigma,'model':key,'registration':reg,'n_correlated_regions':len(subset)}
                for measure in ('psnr','ssim','edge_rmse'):
                    row[measure+'_mean']=round(float(np.mean([r[measure] for r in subset])),6)
                summary.append(row)
    with (output/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    figure.savefig(output/'registered_comparison.png',dpi=130)
    plt.close(figure)
    meta={'dataset':'EMPIAR-10311 local TIFF crops','slice_centers':[13,14],
          'neighbor_slices_all_heldout':True,'alignment':'integer only; nearest-neighbor translation, no wraparound',
          'alignment_estimated_from':'same synthetic-noise input; Gaussian-blurred crops for shift estimation',
          'device':str(device),'crop':args.crop,'regions_per_slice':args.regions,'sigmas':args.sigmas,
          'shift_radius':args.shift_radius,'groups':args.groups,
          'limitations':['No independent clean reference; scores measure fidelity to native noisy image.',
          'Alignment uses information from the noisy central input; it is not a full blind-spot proof.',
          'Small local crops only; no global deformation registration.',
          'V0 inference unmasked; V1 inference masked; comparisons within each checkpoint isolate alignment only.',
          'Training used unregistered sections; registering only at inference introduces a distribution shift.',
          'A higher correlation does not prove anatomical correspondence; poorly correlated neighbors may be biological changes.',
          'Adjacent held-out sections are correlated rather than independent biological replicates.']}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('\nMEAN METRICS (native reference is noisy, NOT ground truth):')
    for row in summary:
        print(f"sigma={row['sigma']:.2f} {row['model']:<12} {row['registration']:<11} PSNR={row['psnr_mean']:.2f} "
              f"SSIM={row['ssim_mean']:.4f} edgeRMSE={row['edge_rmse_mean']:.4f}",flush=True)
    print('Saved:',output/'summary.csv',output/'alignment.csv',output/'registered_comparison.png',flush=True)

if __name__=='__main__': main()
