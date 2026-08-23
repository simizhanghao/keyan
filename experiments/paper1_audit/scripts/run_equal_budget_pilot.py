#!/usr/bin/env python3
"""Equal-budget B1 vs audited C' pilot on fixed P1; not a paper result."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier, RFPatchEmbedder

KEYS = ("iq", "fft", "amp_phase", "oob")

def load(paths, fs=1e6, bw=125e3):
    all_views=[[] for _ in range(4)]; labels=[]; raw=[]
    for p in paths:
        z=np.load(p); data=z['data'].astype(np.float32); y=z['label'].reshape(-1).astype(np.int64)-31
        t=data.shape[1]//2; iq=data[:,:t]+1j*data[:,t:]
        iq=iq/(np.sqrt(np.mean(np.abs(iq)**2,axis=1,keepdims=True))+1e-8); raw.append(np.stack([iq.real,iq.imag],1))
        spec=np.fft.fftshift(np.fft.fft(iq,axis=1),axes=1); f=np.fft.fftshift(np.fft.fftfreq(t,d=1/fs)); mask=np.abs(f)>bw/2
        views=(np.stack([iq.real,iq.imag],1),np.stack([spec.real,spec.imag],1),np.stack([np.abs(iq),np.angle(iq)],1),np.abs(spec[:,mask])[:,None,:])
        for i in range(4): all_views[i].append(views[i])
        labels.append(y)
    return tuple(torch.from_numpy(np.concatenate(x)).float() for x in all_views), torch.from_numpy(np.concatenate(labels)).long(), torch.from_numpy(np.concatenate(raw)).float()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0); paths=sorted(a.source_root.glob('*.npz')); train=[x for x in paths if x.stem!='rtl_2_train']; test=[x for x in paths if x.stem=='rtl_2_train']
    tx,ty,triq=load(train); vx,vy,vriq=load(test); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    b1=MultiViewLateFusionCNN(10).to(dev); emb=RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=True,oob_fusion_type='cross_attn_oob',use_oob_cross_attention=True,fft_norm='log_zscore',oob_norm='ratio'); cp=DeviceClassifier(emb,10,dim=64,depth=2,use_chirp_embedding=False).to(dev)
    results={}
    for name,model in [('B1',b1),('Cprime',cp)]:
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4); ds=TensorDataset(*tx,triq,ty); loader=DataLoader(ds,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(0)); losses=[]; model.train()
        for *items,y in loader:
            if name=='B1': batch={k:v.to(dev) for k,v in zip(KEYS,items[:4])}; logits=model(batch)['logits']
            else: logits=model(items[4].to(dev))['logits']
            loss=F.cross_entropy(logits,y.to(dev)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            def acc(x,iq,y):
                logits=(model({k:v.to(dev) for k,v in zip(KEYS,x)}) if name=='B1' else model(iq.to(dev)))['logits']; return float((logits.argmax(-1).cpu()==y).float().mean())
            results[name]={'parameters':sum(v.numel() for v in model.parameters()),'loss_first':losses[0],'loss_last':losses[-1],'source_acc':acc(tx,triq,ty),'heldout_acc_pilot_only':acc(vx,vriq,vy)}
    payload={'protocol':{'fold':'P1','heldout':'rtl_2','seed':0,'epochs':1,'batch_size':8,'lr':1e-3,'weight_decay':5e-4,'checkpoint_selection':'source_only','blind_opened':False,'training':True},'results':results,'note':'equal-budget pilot only; no paper metric'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
