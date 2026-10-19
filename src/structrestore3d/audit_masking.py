#!/usr/bin/env python3
"""Read-only audit of StructRestore-3D masking, split, and neighbor dependence.
Run from project root. Does not train or overwrite checkpoints.
"""
import argparse, csv, json, sys, random
from pathlib import Path
import numpy as np
import tifffile
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/raw')
    ap.add_argument('--script',default='scripts/structrestore_25d.py')
    ap.add_argument('--checkpoint',default='results/25d_gated/model.pt')
    ap.add_argument('--patch',type=int,default=128)
    ap.add_argument('--samples',type=int,default=100)
    ap.add_argument('--out',default='results/masking_audit')
    a=ap.parse_args()
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(a.input).glob('slice_*.tif'))
    assert len(files)>=7, 'Expected at least seven TIFF slices'
    train,test=files[:-4],files[-4:]
    print('Training slices:',[f.name for f in train])
    print('Held-out slices:',[f.name for f in test])
    print('Filename overlap:',bool(set(train)&set(test)))
    print('Adjacent train/test boundary:',train[-1].name,'/',test[0].name)
    print('CAUTION: Adjacent sections are not biologically independent.')
    print('CAUTION: Noisy neighboring sections can reveal central pixel structure; not a strict blind-spot model.')
    import importlib.util
    spec=importlib.util.spec_from_file_location('audit_model',a.script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=module.Net25D(ck['mode'],ck['width'])
    model.load_state_dict(ck['state_dict']);model.eval()
    print('Checkpoint mode:',ck['mode'],'width:',ck['width'])
    # Check center masking: selected target pixels must differ from corrupted center in most cases.
    ims=[tifffile.memmap(str(f)) for f in train]
    rows=[]
    with torch.inference_mode():
        for i in range(a.samples):
            z=random.randrange(1,len(ims)-1)
            h,w=ims[z].shape
            y=random.randrange(h-a.patch+1);x=random.randrange(w-a.patch+1)
            arr=np.stack([np.asarray(ims[j][y:y+a.patch,x:x+a.patch],dtype=np.float32)/255
                          for j in (z-1,z,z+1)])
            tensor=torch.from_numpy(arr.copy())[None]
            corrupted,target,mask=module.corrupt_center(tensor)
            m=mask.numpy()[0,0]
            original=target.numpy()[0,0]
            changed=(corrupted[:,1:2].numpy()[0,0]!=original)
            selected=int(m.sum())
            changed_count=int((changed&m).sum())
            rows.append(dict(sample=i,z=z,masked=selected,changed=changed_count,
                             masked_unchanged=selected-changed_count,
                             patch_std=float(original.std()),
                             patch_mean=float(original.mean()),
                             masked_target_var=float(original[m].var()) if selected else 0.0))
    with (out/'patch_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    total_mask=sum(r['masked'] for r in rows)
    unchanged=sum(r['masked_unchanged'] for r in rows)
    print(f'Masked target pixels unchanged after replacement: {unchanged}/{total_mask} ({unchanged/max(1,total_mask):.2%})')
    print('Patch std: min/median/max:',*[round(v,5) for v in
          (min(r['patch_std'] for r in rows),np.median([r['patch_std'] for r in rows]),max(r['patch_std'] for r in rows))])
    # Counterfactual probe: keep corrupted center fixed, change only neighboring channels.
    # Measures neighbor dependence, NOT proof of leakage or restoration quality.
    z=5;h,w=ims[z].shape;y=(h-a.patch)//2;x=(w-a.patch)//2
    arr=np.stack([np.asarray(ims[j][y:y+a.patch,x:x+a.patch],dtype=np.float32)/255
                  for j in (z-1,z,z+1)])
    base=torch.from_numpy(arr.copy())[None]
    corrupted,target,mask=module.corrupt_center(base)
    with torch.inference_mode():
        baseline=model(corrupted)
        zero=corrupted.clone();zero[:,0]=0;zero[:,2]=0
        same=corrupted.clone();same[:,0]=same[:,1];same[:,2]=same[:,1]
        swap=corrupted.clone();swap[:,0],swap[:,2]=corrupted[:,2].clone(),corrupted[:,0].clone()
        delta_zero=(model(zero)-baseline).abs().mean().item()
        delta_same=(model(same)-baseline).abs().mean().item()
        delta_swap=(model(swap)-baseline).abs().mean().item()
    print('Neighbor counterfactual mean |output change|:')
    print('  zero neighbors:',round(delta_zero,6))
    print('  copy corrupted center into neighbors:',round(delta_same,6))
    print('  swap neighbors:',round(delta_swap,6))
    report={
      'checkpoint':str(a.checkpoint),'mode':ck['mode'],
      'train_files':[f.name for f in train],'heldout_files':[f.name for f in test],
      'filename_overlap':bool(set(train)&set(test)),
      'masked_pixels':total_mask,'masked_pixels_unchanged':unchanged,
      'masked_unchanged_fraction':unchanged/max(1,total_mask),
      'patch_std_min':min(r['patch_std'] for r in rows),
      'patch_std_median':float(np.median([r['patch_std'] for r in rows])),
      'neighbor_probe_mean_abs_change':{'zero':delta_zero,'copy_center':delta_same,'swap':delta_swap},
      'limitations':[
       'This is a code and behavior audit, not proof of no information leakage.',
       'Adjacent slice channels remain unmasked; correlated neighboring pixels can reveal central structure.',
       'Random neighboring-pixel replacement can equal the target by chance.',
       'Training/test adjacent sections are not independent specimens.',
       'No retraining or performance assessment is performed.'
      ]}
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print('Saved:',out/'audit.json',out/'patch_audit.csv')
if __name__=='__main__': main()
