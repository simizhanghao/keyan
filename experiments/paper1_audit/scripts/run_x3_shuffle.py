#!/usr/bin/env python3
"""X3-B frozen-checkpoint same-receiver cross-device OOB shuffle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier, RFPatchEmbedder

class PairSet(Dataset):
    def __init__(self, paths, indices, seed):
        self.items=[]; self.handles={}; self.labels={}; rng=np.random.default_rng(seed)
        for p in paths:
            with h5py.File(p,'r') as f: ys=np.asarray(f['label'][0],dtype=np.int64)-31
            by={int(y):np.flatnonzero(ys==y) for y in np.unique(ys)}
            for i in indices[p.name]:
                y=int(ys[i]); other=np.concatenate([v for k,v in by.items() if k!=y]); j=int(other[rng.integers(len(other))])
                self.items.append((p,i,j,y))
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        p,i,j,y=self.items[k]
        if p not in self.handles: self.handles[p]=h5py.File(p,'r')
        raw=np.asarray(self.handles[p]['data'][i],dtype=np.float32); raw2=np.asarray(self.handles[p]['data'][j],dtype=np.float32)
        t=raw.shape[0]//2; z=raw[:t]+1j*raw[t:]; z=z/(np.sqrt(np.mean(np.abs(z)**2))+1e-8)
        z2=raw2[:t]+1j*raw2[t:]; z2=z2/(np.sqrt(np.mean(np.abs(z2)**2))+1e-8)
        return torch.from_numpy(np.stack([z.real,z.imag])).float(), torch.from_numpy(np.stack([z2.real,z2.imag])).float(), y

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--fold',required=True); p.add_argument('--model',choices=['B1','Cprime'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--split',choices=['heldout','source_val'],default='heldout'); p.add_argument('--out',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
 dev=torch.device(a.device); paths=sorted(a.data_root.glob('*_train.h5')); hold=[x for x in paths if x.stem==a.fold+'_train'][0]
 if a.split=='heldout': eval_paths=[hold]; idx={hold.name:list(range(8000))}
 else:
  eval_paths=[x for x in paths if x!=hold]; idx={}
  for pth in eval_paths:
   with h5py.File(pth,'r') as f: n=f['data'].shape[0]
   rng=np.random.default_rng(a.seed+sum(pth.name.encode('ascii'))); idx[pth.name]=rng.permutation(n)[int(.9*n):].tolist()
 ds=PairSet(eval_paths,idx,a.seed+9173); dl=DataLoader(ds,batch_size=128,shuffle=False,num_workers=2)
 if a.model=='B1': model=MultiViewLateFusionCNN(10).to(dev)
 else:
  e=RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=True,oob_fusion_type='cross_attn_oob',use_oob_cross_attention=True,fft_norm='log_zscore',oob_norm='ratio'); model=DeviceClassifier(e,10,dim=64,depth=2,use_chirp_embedding=False).to(dev)
 ck=torch.load(a.root/a.fold/a.model/f'seed_{a.seed}'/'best.pt',map_location=dev); model.load_state_dict(ck['model']); model.eval(); correct=total=0; f= torch.fft.fftshift(torch.fft.fftfreq(8192,d=1e-6)).to(dev); mask=f.abs()>62500
 with torch.no_grad():
  for iq,donor,y in dl:
   iq,donor,y=iq.to(dev),donor.to(dev),y.to(dev); z=torch.complex(iq[:,0],iq[:,1]); zd=torch.complex(donor[:,0],donor[:,1]); s=torch.fft.fftshift(torch.fft.fft(z,dim=-1),dim=-1); sd=torch.fft.fftshift(torch.fft.fft(zd,dim=-1),dim=-1); s2=s.clone(); s2[:,mask]=sd[:,mask]
   if a.model=='B1': out=model({'iq':iq,'fft':torch.stack([s2.real,s2.imag],1),'amp_phase':torch.stack([z.abs(),z.angle()],1),'oob':s2.abs()[:,mask].unsqueeze(1)})['logits']
   else:
    iq2=torch.fft.ifft(torch.fft.ifftshift(s2,dim=-1),dim=-1); out=model(iq,oob_iq=torch.stack([iq2.real,iq2.imag],1))['logits']
   correct+=(out.argmax(1)==y).sum().item(); total+=len(y)
 payload={'protocol':{'fold':a.fold,'model':a.model,'seed':a.seed,'split':a.split,'intervention':'same_receiver_cross_device_oob_shuffle','training':False,'blind_opened':False},'accuracy':correct/total,'n_eval':total}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__': main()
