#!/usr/bin/env python3
"""Shen-style RA/CIS PyTorch adaptation under receiver-held-out data."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from run_x2_formal import FOLDS

class RxSet(Dataset):
    def __init__(self, paths, indices, rx_map): self.items=[(p,i,rx_map[p.name]) for p in paths for i in indices[p.name]]; self.h={}
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        p,i,r=self.items[k]
        if p not in self.h: self.h[p]=h5py.File(p,'r')
        raw=np.asarray(self.h[p]['data'][i],dtype=np.float32); y=int(self.h[p]['label'][0,i])-31; t=raw.shape[0]//2; z=raw[:t]+1j*raw[t:]; z=z/(np.sqrt(np.mean(np.abs(z)**2))+1e-8)
        return torch.from_numpy(np.stack([z.real,z.imag])).float(),y,r

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): return x
    @staticmethod
    def backward(ctx,g): return -g

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.c=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(),nn.Conv2d(cout,cout,3,padding=1)); self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x): return torch.relu(self.c(x)+self.skip(x))

class ShenNet(nn.Module):
    def __init__(self,rx,adv):
        super().__init__(); self.adv=adv; self.stem=nn.Conv2d(1,32,7,stride=2,padding=3); self.b1=Block(32,32); self.b2=Block(32,32); self.b3=Block(32,64); self.b4=Block(64,64); self.pool=nn.AvgPool2d(2); self.fc=nn.Linear(64*13*31,512); self.tx=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,10)); self.rx=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,rx))
    def forward(self,x):
        z=torch.relu(self.stem(x)); z=self.b4(self.b3(self.b2(self.b1(z)))); z=self.pool(z).flatten(1); z=torch.nn.functional.normalize(self.fc(z),dim=1); tx=self.tx(z); return (tx,self.rx(GRL.apply(z))) if self.adv else (tx,None)

def cis(iq):
    z=torch.complex(iq[:,0],iq[:,1]); s=torch.stft(z,n_fft=128,hop_length=64,win_length=128,window=torch.ones(128,device=iq.device),center=False,return_complex=True); s=torch.fft.fftshift(s,dim=1); d=s[:,:,1:]/(s[:,:,:-1]+1e-8); x=torch.log10(d.abs().square()+1e-12)[:,19:71,:]; return x.unsqueeze(1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--fold',choices=FOLDS,required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--arm',choices=['Shen-CIS','Shen-RA'],default='Shen-RA'); p.add_argument('--gpu',type=int,default=0); p.add_argument('--epochs',type=int,default=500); p.add_argument('--patience',type=int,default=20); a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.set_device(a.gpu); dev=torch.device('cuda',a.gpu)
    paths=sorted(a.data_root.glob('*_train.h5')); hold=next(q for q in paths if q.stem==a.fold+'_train'); source=[q for q in paths if q!=hold]; rx_map={q.name:i for i,q in enumerate(source)}; idx={}; val={}
    for q in source+[hold]:
        with h5py.File(q,'r') as f:n=f['data'].shape[0]
        perm=np.random.default_rng(a.seed+sum(q.name.encode())).permutation(n); cut=int(.9*n); idx[q.name]=perm[:cut].tolist(); val[q.name]=perm[cut:].tolist()
    tr=DataLoader(RxSet(source,idx,rx_map),64,shuffle=True,num_workers=2,pin_memory=True); val_loader=DataLoader(RxSet(source,val,rx_map),64,shuffle=False,num_workers=2,pin_memory=True); test=DataLoader(RxSet([hold],{hold.name:list(range(n))},{hold.name:0}),64,shuffle=False,num_workers=2)
    model=ShenNet(len(source),a.arm=='Shen-RA').to(dev); opt=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=.9); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.2,patience=10); a.out.parent.mkdir(parents=True,exist_ok=True); w=SummaryWriter(str(a.out.with_suffix('')/'tensorboard')); best=-1; stale=0; hist=[]
    def epoch(dl,train):
        model.train(train); ct=tt=0; loss_sum=0; cm=torch.zeros(10,10,dtype=torch.long)
        for iq,y,r in dl:
            iq,y,r=iq.to(dev),y.to(dev),r.to(dev); tx,rx=model(cis(iq)); loss=nn.functional.cross_entropy(tx,y)+(nn.functional.cross_entropy(rx,r) if rx is not None else 0)
            if train: opt.zero_grad(); loss.backward(); opt.step()
            pred=tx.argmax(1); ct+=(pred==y).sum().item(); tt+=len(y); loss_sum+=float(loss.detach()); cm+=torch.bincount((y*10+pred).detach().cpu(),minlength=100).reshape(10,10)
        tp=cm.diag().float(); denom=cm.sum(1).float()+cm.sum(0).float(); f1=torch.where(denom>0,2*tp/denom,torch.zeros_like(tp)).mean().item()
        return loss_sum/max(1,len(dl)),ct/tt,f1
    for ep in range(1,a.epochs+1):
        tl,ta,_=epoch(tr,True); vl,val_acc,val_f1=epoch(val_loader,False); sched.step(vl); w.add_scalars('accuracy',{'train':ta,'source_val':val_acc},ep); w.add_scalar('macro_f1/source_val',val_f1,ep); w.add_scalar('learning_rate',opt.param_groups[0]['lr'],ep); hist.append({'epoch':ep,'train_acc':ta,'source_val_acc':val_acc,'source_val_macro_f1':val_f1,'source_val_loss':vl});
        if val_acc>best: best=val_acc; stale=0; torch.save({'model':model.state_dict(),'epoch':ep},a.out.with_suffix('.pt'))
        else: stale+=1
        if stale>=a.patience: break
    ck=torch.load(a.out.with_suffix('.pt'),map_location=dev); model.load_state_dict(ck['model']); _,ea,ef1=epoch(test,False); w.close(); payload={'protocol':{'fold':a.fold,'seed':a.seed,'arm':a.arm,'representation':'CIS','receiver_labels_used':a.arm=='Shen-RA','blind_opened':False,'optimizer':'SGD','learning_rate':1e-3,'momentum':.9,'batch_size':64,'checkpoint_selection':'source_validation_accuracy_only','max_epochs':a.epochs,'early_stopping_patience':a.patience,'scheduler':'ReduceLROnPlateau factor=.2 patience=10'},'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':ck['epoch'],'heldout_accuracy':ea,'heldout_macro_f1':ef1,'history':hist,'tensorboard':str(a.out.with_suffix('')/'tensorboard')}; a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__': main()
