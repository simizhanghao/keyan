#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import h5py,numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier,RFPatchEmbedder
from run_x2_formal import views,macro
class Set(Dataset):
 def __init__(self,paths): self.items=[(p,i) for p in paths for i in range(h5py.File(p,'r')['data'].shape[0])]; self.h={}
 def __len__(self): return len(self.items)
 def __getitem__(self,k):
  p,i=self.items[k]
  if p not in self.h:self.h[p]=h5py.File(p,'r')
  x=np.asarray(self.h[p]['data'][i],np.float32); y=int(self.h[p]['label'][0,i])-31; t=x.shape[0]//2; z=x[:t]+1j*x[t:]; z=z/(np.sqrt(np.mean(np.abs(z)**2))+1e-8); return torch.from_numpy(np.stack([z.real,z.imag])).float(),y
def band(x):
 z=torch.complex(x[:,0],x[:,1]); s=torch.fft.fftshift(torch.fft.fft(z),-1); f=torch.fft.fftshift(torch.fft.fftfreq(x.shape[-1],d=1e-6)).to(x.device); q=torch.fft.ifft(torch.fft.ifftshift(s*(f.abs()<=62500),-1)); return torch.stack([q.real,q.imag],1)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['B1-OOB','Cprime-OOB','Cprime-TrueIB'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--epochs',type=int,required=True);a=p.parse_args(); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.set_device(a.gpu);d=torch.device('cuda',a.gpu)
 paths=sorted(a.data_root.glob('*_train.h5')); dl=DataLoader(Set(paths),64,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=4); isb=a.model=='B1-OOB'; oob=a.model!='Cprime-TrueIB'; m=MultiViewLateFusionCNN(10).to(d) if isb else DeviceClassifier(RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=oob,oob_fusion_type='cross_attn_oob' if oob else 'no_oob',use_oob_cross_attention=oob,fft_norm='log_zscore',oob_norm='ratio' if oob else 'none'),10,dim=64,depth=2,use_chirp_embedding=False).to(d); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=5e-4); a.out.parent.mkdir(parents=True,exist_ok=True);w=SummaryWriter(str(a.out.with_suffix('')/'tensorboard'));h=[]
 for e in range(1,a.epochs+1):
  m.train(); ps=[];ys=[];ls=[]
  for x,y in dl:
   x,y=x.to(d,non_blocking=True),y.to(d); x=band(x) if not isb and not oob else x; z=m(views(x))['logits'] if isb else m(x)['logits']; loss=F.cross_entropy(z,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ps+=z.argmax(1).detach().cpu().tolist();ys+=y.cpu().tolist();ls.append(float(loss.detach()))
  acc=float(np.mean(np.array(ps)==np.array(ys)));f=macro(np.array(ps),np.array(ys));h.append({'epoch':e,'train_loss':float(np.mean(ls)),'train_acc':acc,'train_macro_f1':f});w.add_scalar('train/accuracy',acc,e);w.add_scalar('train/macro_f1',f,e)
 w.close(); ckpt=a.out.with_suffix('.pt'); torch.save({'model':m.state_dict(),'epoch':a.epochs,'model_name':a.model,'seed':a.seed},ckpt); a.out.write_text(json.dumps({'protocol':{'model':a.model,'seed':a.seed,'epochs':a.epochs,'batch_size':64,'receivers':14,'samples_per_epoch':len(dl.dataset),'steps_per_epoch':len(dl),'validation':'none','checkpoint_selection':'none','blind_opened':False,'optimizer':'AdamW','learning_rate':1e-3,'weight_decay':5e-4},'parameters':sum(p.numel() for p in m.parameters()),'history':h,'checkpoint':str(ckpt.resolve()),'tensorboard':str((a.out.with_suffix('')/'tensorboard').resolve())},indent=2)+'\n')
if __name__=='__main__':main()
