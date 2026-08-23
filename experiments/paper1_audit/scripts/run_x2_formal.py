#!/usr/bin/env python3
"""X2 formal B1/C' runner with TensorBoard and source-only validation."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier, RFPatchEmbedder

FOLDS=("rtl_2","rtl_5","b200_1","b200_mini_1","b210_1","pluto_1")
class H5Set(Dataset):
    def __init__(self, paths, indices): self.items=[(p,i) for p in paths for i in indices[p.name]]; self.handles={}
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        p,i=self.items[k]
        if p not in self.handles: self.handles[p]=h5py.File(p,'r')
        f=self.handles[p]; raw=np.asarray(f['data'][i],dtype=np.float32); y=int(np.asarray(f['label'][:, i]).reshape(-1)[0])-31
        t=raw.shape[0]//2; iq=raw[:t]+1j*raw[t:]; iq=iq/(np.sqrt(np.mean(np.abs(iq)**2))+1e-8)
        return torch.from_numpy(np.stack([iq.real,iq.imag])).float(),y
def macro(pred,y,n=10):
    vals=[]
    for c in range(n):
        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum(); vals.append((2*tp)/(2*tp+fp+fn+1e-8))
    return float(np.mean(vals))
def views(iq, fs=1e6,bw=125e3):
    z=torch.complex(iq[:,0],iq[:,1]); spec=torch.fft.fftshift(torch.fft.fft(z,dim=-1),dim=-1); f=torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1],d=1/fs)).to(iq.device); m=f.abs()>bw/2
    return {'iq':iq,'fft':torch.stack([spec.real,spec.imag],1),'amp_phase':torch.stack([z.abs(),z.angle()],1),'oob':spec.abs()[:,m].unsqueeze(1)}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--out-root',type=Path,required=True); p.add_argument('--fold',required=True,choices=FOLDS); p.add_argument('--model',choices=('B1','Cprime'),required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--gpu',type=int,required=True); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.set_device(a.gpu); dev=torch.device('cuda',a.gpu)
 paths=sorted(a.source_root.glob('*_train.h5')); hold=[p for p in paths if p.stem==a.fold+'_train'][0]; source=[p for p in paths if p!=hold]
 idx={}; val_idx={}
 for pth in source+[hold]:
  with h5py.File(pth,'r') as f: n=f['data'].shape[0]
  allidx=np.arange(n); rng=np.random.default_rng(a.seed+hash(pth.name)%10000); rng.shuffle(allidx); cut=int(.9*n); idx[pth.name]=allidx[:cut].tolist(); val_idx[pth.name]=allidx[cut:].tolist()
 train=H5Set(source,idx); val=H5Set(source,val_idx); test=H5Set([hold],{hold.name:list(range(n))})
 tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True); va=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=2,pin_memory=True); te=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=2,pin_memory=True)
 if a.model=='B1': model=MultiViewLateFusionCNN(10).to(dev)
 else:
  e=RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=True,oob_fusion_type='cross_attn_oob',use_oob_cross_attention=True,fft_norm='log_zscore',oob_norm='ratio'); model=DeviceClassifier(e,10,dim=64,depth=2,use_chirp_embedding=False).to(dev)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4); run=a.out_root/a.fold/a.model/f'seed_{a.seed}'; run.mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(run/'tensorboard')); best=-1; history=[]
 def epoch(loader,train_mode):
  model.train(train_mode); total=correct=0; ls=[]; ps=[]; ys=[]
  for iq,y in loader:
   iq=iq.to(dev,non_blocking=True); y=y.to(dev)
   if a.model=='B1': out=model(views(iq))['logits']
   else: out=model(iq)['logits']
   loss=F.cross_entropy(out,y)
   if train_mode: opt.zero_grad(); loss.backward(); opt.step()
   ls.append(float(loss.detach().cpu())); pred=out.argmax(1); correct+=(pred==y).sum().item(); total+=len(y); ps.extend(pred.detach().cpu().numpy()); ys.extend(y.detach().cpu().numpy())
  return float(np.mean(ls)),correct/total,macro(np.array(ps),np.array(ys))
 for ep in range(1,a.epochs+1):
  tl,ta,tf=epoch(tr,True); vl,va_,vf=epoch(va,False); writer.add_scalars('loss',{'train':tl,'source_val':vl},ep); writer.add_scalars('accuracy',{'train':ta,'source_val':va_},ep); writer.add_scalars('macro_f1',{'train':tf,'source_val':vf},ep); history.append({'epoch':ep,'train_loss':tl,'source_val_loss':vl,'train_acc':ta,'source_val_acc':va_,'source_val_macro_f1':vf});
  if va_>best: best=va_; torch.save({'model':model.state_dict(),'epoch':ep},run/'best.pt')
 ck=torch.load(run/'best.pt',map_location=dev); model.load_state_dict(ck['model']); el,ea,ef=epoch(te,False); writer.add_scalar('heldout/accuracy',ea,ck['epoch']); writer.add_scalar('heldout/macro_f1',ef,ck['epoch']); writer.close(); payload={'protocol':{'fold':a.fold,'model':a.model,'seed':a.seed,'gpu':a.gpu,'epochs':a.epochs,'checkpoint_selection':'source_validation_only','blind_opened':False},'parameters':sum(x.numel() for x in model.parameters()),'source_train_packets':len(train),'source_val_packets':len(val),'heldout_packets':len(test),'best_epoch':ck['epoch'],'history':history,'heldout_accuracy':ea,'heldout_macro_f1':ef,'tensorboard':str((run/'tensorboard').resolve())}; (run/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__': main()
