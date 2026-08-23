#!/usr/bin/env python3
"""X3-E grouped frozen-embedding device/receiver probes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data import DataLoader
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier,RFPatchEmbedder
from run_x2_formal import H5Set,views

def linprobe(x,y,nc,tr,te):
 mu=x[tr].mean(0,keepdim=True); sd=x[tr].std(0,keepdim=True).clamp_min(1e-5); x=(x-mu)/sd
 w=torch.zeros(x.shape[1],nc,requires_grad=True); b=torch.zeros(nc,requires_grad=True); opt=torch.optim.Adam([w,b],lr=.05,weight_decay=1e-3)
 for _ in range(300):
  loss=torch.nn.functional.cross_entropy(x[tr]@w+b,y[tr]); opt.zero_grad(); loss.backward(); opt.step()
 return float(((x[te]@w+b).argmax(1)==y[te]).float().mean())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--fold',required=True); p.add_argument('--model',choices=['B1','Cprime'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); dev=torch.device(a.device)
 paths=sorted(a.data_root.glob('*_train.h5')); hold=[x for x in paths if x.stem==a.fold+'_train'][0]; ep=[x for x in paths if x!=hold]; items=[]
 for pth in ep:
  with h5py.File(pth,'r') as f:
   ys=np.asarray(f['label'][0],dtype=np.int64)-31
  for d in range(10):
   ii=np.flatnonzero(ys==d)[:256]
   for i in ii: items.append((pth,int(i),d))
 class DS(torch.utils.data.Dataset):
  def __len__(self): return len(items)
  def __getitem__(self,k):
   pth,i,d=items[k]
   if not hasattr(self,'hs'): self.hs={}
   if pth not in self.hs:self.hs[pth]=h5py.File(pth,'r')
   raw=np.asarray(self.hs[pth]['data'][i],dtype=np.float32); t=raw.shape[0]//2; z=raw[:t]+1j*raw[t:]; z=z/(np.sqrt(np.mean(np.abs(z)**2))+1e-8); return torch.from_numpy(np.stack([z.real,z.imag])).float(),d
 dl=DataLoader(DS(),batch_size=128,shuffle=False,num_workers=2); groups=[]
 if a.model=='B1': model=MultiViewLateFusionCNN(10).to(dev)
 else:
  e=RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=True,oob_fusion_type='cross_attn_oob',use_oob_cross_attention=True,fft_norm='log_zscore',oob_norm='ratio'); model=DeviceClassifier(e,10,dim=64,depth=2,use_chirp_embedding=False).to(dev)
 model.load_state_dict(torch.load(a.root/a.fold/a.model/f'seed_{a.seed}'/'best.pt',map_location=dev)['model']); model.eval(); feats=[]; labels=[]
 with torch.no_grad():
  for iq,y in dl:
   iq=iq.to(dev); out=model(views(iq),return_features=True) if a.model=='B1' else model(iq,return_features=True); feats.append(out['features'].detach().cpu()); labels.extend(y.tolist())
 x=torch.cat(feats); labels=np.asarray(labels); cell=[]; recv=[]; device=[]; start=0
 for ri,pth in enumerate(ep):
  for d in range(10): cell.append(x[start:start+256].mean(0)); recv.append(ri); device.append(d); start+=256
 x=torch.stack(cell); recv=torch.tensor(recv); device=torch.tensor(device); g=torch.Generator().manual_seed(a.seed+11); rp=torch.randperm(len(ep),generator=g); rcut=max(1,int(.75*len(ep))); rtr=set(rp[:rcut].tolist()); rmask=torch.tensor([int(i) in rtr for i in recv.tolist()]); device_acc=linprobe(x,device,10,rmask,~rmask)
 # Receiver probe holds out devices, so every receiver class is seen in train.
 dp=torch.randperm(10,generator=g); dtr=set(dp[:7].tolist()); dmask=torch.tensor([int(i) in dtr for i in device.tolist()]);
 mu=x[dmask].mean(0,keepdim=True); sd=x[dmask].std(0,keepdim=True).clamp_min(1e-5); xn=(x-mu)/sd
 w=torch.zeros(x.shape[1],len(ep),requires_grad=True); b=torch.zeros(len(ep),requires_grad=True); opt=torch.optim.Adam([w,b],lr=.05,weight_decay=1e-3); tr=dmask; te=~dmask
 for _ in range(300):
  loss=torch.nn.functional.cross_entropy(xn[tr]@w+b,recv[tr]); opt.zero_grad(); loss.backward(); opt.step()
 receiver_acc=float(((xn[te]@w+b).argmax(1)==recv[te]).float().mean())
 payload={'protocol':{'fold':a.fold,'model':a.model,'seed':a.seed,'training_backbone':False,'group_unit':'receiver_device_cell_mean_256_packets','blind_opened':False},'device_probe_accuracy':device_acc,'receiver_probe_accuracy':receiver_acc,'n_cells':len(cell)}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__':main()
