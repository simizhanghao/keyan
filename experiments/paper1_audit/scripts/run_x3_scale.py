#!/usr/bin/env python3
"""X3-A frozen-checkpoint OOB scale sensitivity; no training/blind RX."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import DataLoader
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from run_x2_formal import H5Set

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--data-root', type=Path, required=True); p.add_argument('--fold', required=True)
    p.add_argument('--model', required=True, choices=['B1','Cprime']); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--split', choices=['heldout','source_val'], default='heldout')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    dev = torch.device(a.device); paths = sorted(a.data_root.glob('*_train.h5'))
    hold = [x for x in paths if x.stem == a.fold + '_train'][0]; idx = {}
    if a.split == 'heldout':
        with h5py.File(hold, 'r') as f: n = f['data'].shape[0]
        idx[hold.name] = list(range(n)); eval_paths = [hold]
    else:
        eval_paths = [x for x in paths if x != hold]
        for pth in eval_paths:
            with h5py.File(pth, 'r') as f: n = f['data'].shape[0]
            # Stable source-validation partition, independent of Python hash randomization.
            rng = np.random.default_rng(a.seed + sum(pth.name.encode('ascii')))
            perm = rng.permutation(n); idx[pth.name] = perm[int(0.9*n):].tolist()
    dl = DataLoader(H5Set(eval_paths, idx), batch_size=128, shuffle=False, num_workers=2)
    if a.model == 'B1': model = MultiViewLateFusionCNN(10).to(dev)
    else:
        e = RFPatchEmbedder(window_size=8192, patch_size=256, dim=64, cnn_stem_dim=32, patch_embed_type='cnn_stem', use_oob=True, oob_fusion_type='cross_attn_oob', use_oob_cross_attention=True, fft_norm='log_zscore', oob_norm='ratio')
        model = DeviceClassifier(e, 10, dim=64, depth=2, use_chirp_embedding=False).to(dev)
    ck = torch.load(a.root / a.fold / a.model / f'seed_{a.seed}' / 'best.pt', map_location=dev); model.load_state_dict(ck['model']); model.eval()
    scales = [0.5, 0.70710678, 1.0, 1.41421356, 2.0]; result = []
    with torch.no_grad():
        for scale in scales:
            correct = total = 0
            for iq, y in dl:
                iq, y = iq.to(dev), y.to(dev); z = torch.complex(iq[:,0], iq[:,1]); s = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1); f = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1e-6)).to(dev); mask = f.abs() > 62500; s2 = s.clone(); s2[:, mask] *= scale
                if a.model == 'B1':
                    out = model({'iq': iq, 'fft': torch.stack([s2.real,s2.imag],1), 'amp_phase': torch.stack([z.abs(),z.angle()],1), 'oob': s2.abs()[:,mask].unsqueeze(1)})['logits']
                else:
                    iq2 = torch.fft.ifft(torch.fft.ifftshift(s2, dim=-1), dim=-1); out = model(iq, oob_iq=torch.stack([iq2.real,iq2.imag],1))['logits']
                correct += (out.argmax(1) == y).sum().item(); total += len(y)
            result.append({'scale': scale, 'accuracy': correct / total})
    payload = {'protocol': {'fold': a.fold, 'model': a.model, 'seed': a.seed, 'split': a.split, 'training': False, 'blind_opened': False}, 'curve': result, 'clean_accuracy': next(x['accuracy'] for x in result if x['scale'] == 1.0), 'n_eval': len(dl.dataset)}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(payload, indent=2) + '\n'); print(json.dumps(payload))
if __name__ == '__main__': main()
