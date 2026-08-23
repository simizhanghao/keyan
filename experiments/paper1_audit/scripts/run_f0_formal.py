#!/usr/bin/env python3
"""Formal F0/CT continuation training on the locked X2 HDF5 protocol."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import h5py,numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier,RFPatchEmbedder
from run_x2_formal import H5Set,views,macro
def main():
 p=argparse.ArgumentParser(); p.add_argument('--x2-root',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--out-root',type=Path,required=True); p.add_argument('--fold',required=True); p.add_argument('--model',choices=['B1','Cprime'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--arm',choices=['CT','F0'],required=True); p.add_argument('--gpu',type=int,required=True); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.set_device(a.gpu); dev=torch.device('cuda',a.gpu); paths=sorted(a.data_root.glob('*_train.h5')); hold=[x for x in paths if x.stem==a.fold+'_train'][0]; source=[x for x in paths if x!=hold]; idx={}; vi={}
 for q in source+[hold]:
  with h5py.File(q,'r') as f:n=f['data'].shape[0]
  r=np.random.default_rng(a.seed+sum(q.name.encode('ascii'))); z=r.permutation(n); idx[q.name]=z[:int(.9*n)].tolist(); vi[q.name]=z[int(.9*n):].tolist()
 tr=DataLoader(H5Set(source,idx),batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True); va=DataLoader(H5Set(source,vi),batch_size=a.batch_size,shuffle=False,num_workers=2,pin_memory=True)
 if a.model=='B1': model=MultiViewLateFusionCNN(10).to(dev)
 else:
  e=RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=True,oob_fusion_type='cross_attn_oob',use_oob_cross_attention=True,fft_norm='log_zscore',oob_norm='ratio'); model=DeviceClassifier(e,10,dim=64,depth=2,use_chirp_embedding=False).to(dev)
 init=a.x2_root/a.fold/a.model/f'seed_{a.seed}'/'best.pt'; model.load_state_dict(torch.load(init,map_location=dev)['model']); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4); run=a.out_root/a.fold/a.model/a.arm/f'seed_{a.seed}'; run.mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(run/'tensorboard')); best=-1
 def epoch(dl,train):
  model.train(train); ps=[]; ys=[]; losses=[]
  for iq,y in dl:
   iq,y=iq.to(dev),y.to(dev); out=None
   if a.arm=='F0' and train:
    z=torch.complex(iq[:,0],iq[:,1]); s=torch.fft.fftshift(torch.fft.fft(z,dim=-1),dim=-1); f=torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1],d=1e-6)).to(dev); m=f.abs()>62500; scale=torch.exp(torch.empty(len(iq),device=dev).uniform_(np.log(.5),np.log(2.0))); s[:,m]*=scale[:,None]
    if a.model=='B1': out=model({'iq':iq,'fft':torch.stack([s.real,s.imag],1),'amp_phase':torch.stack([z.abs(),z.angle()],1),'oob':s.abs()[:,m].unsqueeze(1)})['logits']
    else:
     iq2=torch.fft.ifft(torch.fft.ifftshift(s,dim=-1),dim=-1); out=model(iq,oob_iq=torch.stack([iq2.real,iq2.imag],1))['logits']
   else: out=model(views(iq),)['logits'] if a.model=='B1' else model(iq)['logits']
   loss=F.cross_entropy(out,y); 
   if train: opt.zero_grad(); loss.backward(); opt.step()
   losses.append(float(loss.detach())); ps.extend(out.argmax(1).detach().cpu().numpy()); ys.extend(y.cpu().numpy())
  return float(np.mean(losses)),float(np.mean(np.asarray(ps)==np.asarray(ys))),macro(np.asarray(ps),np.asarray(ys))
 for ep in range(1,a.epochs+1):
  tl,ta,tf=epoch(tr,True); vl,va_,vf=epoch(va,False); writer.add_scalars('accuracy',{'train':ta,'source_val':va_},ep); writer.add_scalars('macro_f1',{'train':tf,'source_val':vf},ep)
  if va_>best: best=va_; torch.save({'model':model.state_dict(),'epoch':ep},run/'best.pt')
 writer.close(); payload={'protocol':{'fold':a.fold,'model':a.model,'arm':a.arm,'seed':a.seed,'init_checkpoint':str(init),'checkpoint_selection':'source_validation_only','blind_opened':False},'best_epoch':torch.load(run/'best.pt',map_location='cpu')['epoch'],'tensorboard':str((run/'tensorboard').resolve())}; (run/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__':main()
