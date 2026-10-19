#!/usr/bin/env python3
"""Inference and diagnostic figures for StructRestore-3D TinyCNN smoke-test baseline."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_baseline import TinyCNN


def predict_tiled(model, img, device, tile=256, halo=8):
    h, w = img.shape
    out = np.empty((h, w), dtype=np.float32)
    with torch.inference_mode():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y2, x2 = min(y+tile, h), min(x+tile, w)
                a, b = max(0, y-halo), min(h, y2+halo)
                c, d = max(0, x-halo), min(w, x2+halo)
                inp = torch.from_numpy(img[a:b, c:d].copy())[None, None].to(device)
                pred = model(inp).squeeze().cpu().numpy()
                out[y:y2, x:x2] = pred[y-a:y2-a, x-c:x2-c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/processed')
    ap.add_argument('--checkpoint', default='results/baseline_smoke/baseline.pt')
    ap.add_argument('--out', default='results/baseline_inference')
    ap.add_argument('--tile', type=int, default=256)
    args = ap.parse_args()
    if args.tile < 32:
        ap.error('--tile must be at least 32')
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = TinyCNN(width=int(ckpt['width']))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    inputs = Path(args.input)
    heldout = [inputs / f'slice_{i:04d}.tif' for i in range(12,16)]
    if not all(p.is_file() for p in heldout):
        raise FileNotFoundError('Expected held-out slices 0012–0015 in ' + str(inputs))
    rows=[]
    for path in heldout:
        img_uint8 = tifffile.imread(path)
        if img_uint8.ndim != 2 or img_uint8.dtype != np.uint8:
            raise ValueError(f'Expected grayscale uint8 TIFF: {path}')
        img = img_uint8.astype(np.float32)/255.0
        start = time.perf_counter()
        pred = predict_tiled(model, img, device, tile=args.tile)
        elapsed = time.perf_counter()-start
        pred_clip = np.clip(pred, 0, 1)
        tifffile.imwrite(outdir / f'{path.stem}_denoised.tif', (pred_clip*255+0.5).astype(np.uint8))
        diff = pred_clip-img
        # Show same intensity range for original and prediction, and a signed residual.
        h,w=img.shape; side=min(400,h,w)
        y=(h-side)//2; x=(w-side)//2
        original_crop=img[y:y+side,x:x+side]
        pred_crop=pred_clip[y:y+side,x:x+side]
        diff_crop=diff[y:y+side,x:x+side]
        vmax=max(float(np.quantile(np.abs(diff_crop),0.99)), 0.01)
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        axes[0].imshow(original_crop,cmap='gray',vmin=0,vmax=1)
        axes[0].set_title('Original noisy')
        axes[1].imshow(pred_crop,cmap='gray',vmin=0,vmax=1)
        axes[1].set_title('Baseline output')
        im=axes[2].imshow(diff_crop,cmap='coolwarm',vmin=-vmax,vmax=vmax)
        axes[2].set_title('Output minus input')
        fig.colorbar(im,ax=axes[2],fraction=0.046,pad=0.04)
        for ax in axes: ax.axis('off')
        fig.suptitle(f'{path.name}: held-out, downsampled development data')
        fig.savefig(outdir / f'{path.stem}_comparison.png',dpi=170)
        plt.close(fig)
        mae=float(np.mean(np.abs(diff)))
        rows.append({'file':path.name,'seconds':round(elapsed,3),'mean_abs_change':round(mae,6),'out_of_range_fraction':round(float(np.mean((pred<0)|(pred>1))),6)})
        print(f'{path.name}: {elapsed:.2f}s; mean absolute change={mae:.5f}',flush=True)
    with (outdir/'diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with (outdir/'inference.json').open('w') as f:
        json.dump({'device':str(device),'checkpoint':args.checkpoint,'input':args.input,'note':'No ground truth; change magnitude is NOT a denoising quality metric','results':rows},f,indent=2)
    print('Saved outputs to',outdir)

if __name__=='__main__': main()
