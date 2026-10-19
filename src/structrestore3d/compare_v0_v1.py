#!/usr/bin/env python3
"""Step 27: controlled V0 vs V1 EM FIB-SEM added-noise stress test.

Reference: original *naturally noisy* TIFF, NOT verified clean ground truth.
V1: masked-partition prediction: each pixel is inferred ONLY while concealed
     at the same pixel coordinate in all input sections, avoiding unmasked
     inference mismatch. Masked pixels are given independent uniform filler.
V0: native inference exactly as originally used. Comparisons confound training
     changes and inference protocols; not causal architecture ablations.
No files in existing results directories are written or altered.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loads existing project architectures without modifying source or checkpoints.
SCRIPT_DIR=Path(__file__).resolve().parent
if SCRIPT_DIR.name!='scripts' and (SCRIPT_DIR/'scripts').is_dir():
    SCRIPT_DIR=SCRIPT_DIR/'scripts'
sys.path.insert(0,str(SCRIPT_DIR))
from train_fullres import TinyCNN
from structrestore_25d import Net25D
from train_v1_audited import Model as V1Model

DEFAULT_PATHS={
  'v0_single':'results/baseline_fullres/baseline.pt',
  'v0_plain':'results/25d_plain/model.pt',
  'v0_gated':'results/25d_gated/model.pt',
  'v1_single':'results/v1_single/model.pt',
  'v1_plain':'results/v1_plain/model.pt',
  'v1_gated':'results/v1_gated/model.pt',
}

def load_models(paths,dev):
    models={}
    for key,path in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=True)
        mode=key.split('_')[1]
        if key.startswith('v0'):
            net=TinyCNN(width=int(ck.get('width',32))) if mode=='single' else Net25D(mode,int(ck.get('width',32)))
            if mode!='single' and ck.get('mode')!=mode:raise ValueError(f'V0 mode mismatch: {key}: {ck.get("mode")}')
        else:
            if ck.get('mode')!=mode:raise ValueError(f'V1 mode mismatch: {key}: {ck.get("mode")}')
            net=V1Model(mode,int(ck.get('width',32)))
        net.load_state_dict(ck['state_dict'],strict=True)
        models[key]=net.to(dev).eval()
        print('Loaded',key, '<-',path,flush=True)
    return models

def v0_predict(model,mode,stack,dev,pad):
    arr=stack[1:2] if mode=='single' else stack
    with torch.inference_mode():
        inp=torch.from_numpy(np.ascontiguousarray(arr[None])).to(dev)
        out=model(inp).clamp(0,1)[0,0].cpu().numpy()
    return out[pad:-pad,pad:-pad]

def masked_partition_v1(model,stack,dev,pad,groups,rng):
    """Predict every pixel *once* under a disjoint random ~5% mask partition.

    Independent uniform replacement for each masked channel matches training
    mask value distribution. All channels use exactly the same boolean mask.
    """
    _,h,w=stack.shape
    group_ids=rng.integers(0,groups,size=(h,w),dtype=np.uint8 if groups<=256 else np.int16)
    output=np.empty((h-2*pad,w-2*pad),dtype=np.float32)
    counts=np.zeros(output.shape,dtype=np.uint8)
    base=torch.from_numpy(np.ascontiguousarray(stack[None])).to(dev)
    with torch.inference_mode():
        for g in range(groups):
            selected=(group_ids==g)
            if not np.any(selected[pad:-pad,pad:-pad]):continue
            mask=torch.from_numpy(selected[None,None]).to(dev)
            # Draw filler per channel independently, as in V1 training.
            filler=torch.rand_like(base)
            masked=torch.where(mask.expand_as(base),filler,base)
            prediction=model(masked).clamp(0,1)[0,0].cpu().numpy()[pad:-pad,pad:-pad]
            interior=selected[pad:-pad,pad:-pad]
            output[interior]=prediction[interior]
            counts[interior]+=1
    if not np.all(counts==1):raise RuntimeError('Mask partitions did not cover each interior pixel exactly once')
    return output

def calc_metrics(ref,img):
    img=np.clip(img,0,1).astype(np.float32)
    gy,gx=np.gradient(ref)
    py,px=np.gradient(img)
    return {'psnr':float(peak_signal_noise_ratio(ref,img,data_range=1.0)),
            'ssim':float(structural_similarity(ref,img,data_range=1.0)),
            'edge_rmse':float(np.sqrt(np.mean((gy-py)**2+(gx-px)**2)))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/raw')
    p.add_argument('--out',default='results/v0_v1_comparison')
    p.add_argument('--crop',type=int,default=256,help='Central output crop width; extra 16px context used')
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--sigmas',type=float,nargs='+',default=[0.05,0.10])
    p.add_argument('--groups',type=int,default=20,help='V1 disjoint random mask groups, 20 ~ 5%% masked per pass')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--models-root',default='.',help='Project root containing model checkpoints')
    args=p.parse_args()
    pad=16
    if args.crop<64 or args.trials<1 or args.groups<2 or args.groups>255 or any(not 0<s<0.5 for s in args.sigmas):
        p.error('Require crop>=64, trials>=1, 2<=groups<=255 and 0<sigma<0.5')
    files=sorted(Path(args.input).glob('slice_*.tif'))
    if len(files)!=16:raise ValueError(f'Expected 16 original TIFFs; found {len(files)}')
    paths={k:Path(args.models_root)/v for k,v in DEFAULT_PATHS.items()}
    missing=[str(f) for f in paths.values() if not f.is_file()]
    if missing:raise FileNotFoundError('Missing checkpoints: '+', '.join(missing))
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}. Use new --out to avoid overwriting experiments.')
    out.mkdir(parents=True,exist_ok=True)
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:',dev,flush=True)
    models=load_models(paths,dev)
    rng=np.random.default_rng(args.seed)
    rows=[];figure_done=False
    # Require both context sections to belong to the four held-out slices.
    for z in (13,14):
        ims=[tifffile.memmap(str(files[j])) for j in (z-1,z,z+1)]
        h,w=ims[1].shape;side=args.crop
        if min(h,w)<side+2*pad:raise ValueError('Image too small for crop and 16px halo')
        centers=[(h//4,w//4),(h//2,w//2),(3*h//4,3*w//4)]
        for region,(cy,cx) in enumerate(centers):
            y=max(pad,min(h-side-pad,cy-side//2))
            x=max(pad,min(w-side-pad,cx-side//2))
            stack=np.stack([np.asarray(im[y-pad:y+side+pad,x-pad:x+side+pad],dtype=np.float32)/255 for im in ims])
            reference=stack[1,pad:-pad,pad:-pad].copy()
            for sigma in args.sigmas:
                for trial in range(args.trials):
                    noisy=np.clip(stack+rng.normal(0,sigma,stack.shape).astype(np.float32),0,1)
                    noisy_center=noisy[1,pad:-pad,pad:-pad]
                    candidates={'added_noise':noisy_center,
                                'gaussian_0p8':gaussian_filter(noisy_center,0.8),
                                'median_3x3':median_filter(noisy_center,size=3)}
                    times={k:0.0 for k in candidates}
                    # SHARE one mask assignment and filler across the three V1 networks:
                    # reset torch generator before each call so masks/filler match.
                    # This probe is repeatable on CPU and MPS within each run.
                    rng_state=rng.bit_generator.state
                    for key,model in models.items():
                        start=time.perf_counter()
                        if key.startswith('v0'):
                            candidates[key]=v0_predict(model,key.split('_')[1],noisy,dev,pad)
                        else:
                            rng.bit_generator.state=rng_state
                            torch.manual_seed(args.seed+100000*z+1000*region+100*trial+int(round(100*sigma)))
                            candidates[key]=masked_partition_v1(model,noisy,dev,pad,args.groups,rng)
                        times[key]=time.perf_counter()-start
                    # Advance RNG state by exactly one V1 mask draw for next noise trial.
                    for name,img in candidates.items():
                        m=calc_metrics(reference,img)
                        rows.append({'slice':f'slice_{z:04d}','region':region,'sigma':sigma,
                                     'trial':trial,'method':name,'seconds':round(times[name],5),**m})
                    if not figure_done:
                        fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
                        ordered=[('Reference (native noise)',reference),('Added noise',noisy_center),
                                 ('Gaussian 0.8',candidates['gaussian_0p8']),
                                 ('V0 single',candidates['v0_single']),('V0 plain',candidates['v0_plain']),
                                 ('V0 gated',candidates['v0_gated']),('V1 single',candidates['v1_single']),
                                 ('V1 plain',candidates['v1_plain']),('V1 gated',candidates['v1_gated'])]
                        for ax,(name,img) in zip(axes.ravel(),ordered):
                            ax.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                            ax.set_title(name);ax.axis('off')
                        fig.suptitle(f'Added noise sigma {sigma} | {files[z].stem} | region {region}; reference is NOT clean')
                        fig.savefig(out/'comparison.png',dpi=140)
                        plt.close(fig);figure_done=True
        print('Finished',files[z].stem,flush=True)
    methods=['added_noise','gaussian_0p8','median_3x3']+list(DEFAULT_PATHS)
    with (out/'per_trial.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for sigma in args.sigmas:
        for method in methods:
            subset=[r for r in rows if r['sigma']==sigma and r['method']==method]
            r={'sigma':sigma,'method':method,'n_correlated_evaluations':len(subset)}
            for metric in ['psnr','ssim','edge_rmse','seconds']:
                vals=np.array([v[metric] for v in subset],dtype=np.float64)
                r[metric+'_mean']=round(float(vals.mean()),6)
                r[metric+'_sd']=round(float(vals.std(ddof=1)),6) if len(vals)>1 else 0.
            summary.append(r)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    metadata={'experiment':'V0 vs V1 / added-noise stress test','device':str(dev),'seed':args.seed,
      'train_test_boundary':'train slices 0000-0011, heldout 0012-0015',
      'evaluated_slice_centers':['slice_0013','slice_0014'],
      'slice_context_always_heldout':True,'crop':args.crop,'trials':args.trials,
      'sigmas':args.sigmas,'v1_mask_groups':args.groups,
      'checkpoints':{k:str(v) for k,v in paths.items()},
      'limitations':[
         'Original FIB-SEM reference retains unknown native noise; it is NOT clean ground truth.',
         'Metrics assess fidelity to original after extra simulated noise, not removal of native noise.',
         'V1 inference predicts every pixel once while directly masked using random disjoint partitions; V0 uses native unmasked inference.',
         'V1 changes training masking and patch sampling plus inference procedure: not a causal isolation of any one change.',
         'No independent biological specimens; slices, patches and noise replicates are correlated.',
         'Edge RMSE is gradient fidelity to noisy original, not verified membrane-preservation accuracy.',
         'Models were trained on native noisy images; extra Gaussian noise is an out-of-distribution stress test.'
      ]}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('\nMEAN METRICS: high PSNR/SSIM is better; low edge_RMSE is better',flush=True)
    for r in summary:
        print(f"sigma={r['sigma']:.2f} {r['method']:<14} PSNR={r['psnr_mean']:6.2f}  SSIM={r['ssim_mean']:.4f}  edge_RMSE={r['edge_rmse_mean']:.4f}",flush=True)
    print('\nSaved:',out/'summary.csv',out/'per_trial.csv',out/'comparison.png',out/'metadata.json',flush=True)

if __name__=='__main__':main()
