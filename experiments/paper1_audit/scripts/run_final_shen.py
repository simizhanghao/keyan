#!/usr/bin/env python3
"""Final Shen-style runs: all development receivers, fixed epochs, no blind data."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from run_x4c_shen_port import ShenNet

class CachedCISSet(torch.utils.data.Dataset):
    def __init__(self, paths, cache_root, rx_map):
        self.items=[]; self.cache_root=cache_root; self.rx_map=rx_map; self.arrays={}; self.labels={}
        for p in paths:
            meta=json.loads((cache_root/(p.stem+'.json')).read_text())
            self.items.extend((p.stem,i) for i in range(meta['shape'][0]))
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        stem,i=self.items[k]
        if stem not in self.arrays:
            meta=json.loads((self.cache_root/(stem+'.json')).read_text()); self.arrays[stem]=np.load(meta['cache'],mmap_mode='r'); self.labels[stem]=np.load(meta['labels'],mmap_mode='r')
        return torch.from_numpy(self.arrays[stem][i].copy()), int(self.labels[stem][i]), self.rx_map[stem]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--arm',choices=['Shen-CIS','Shen-RA'],required=True)
    p.add_argument('--gpu',type=int,required=True); p.add_argument('--epochs',type=int,required=True); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.set_device(a.gpu); dev=torch.device('cuda',a.gpu)
    paths=sorted(a.data_root.glob('*_train.h5')); rx_map={q.stem:i for i,q in enumerate(paths)}; cache_root=a.data_root.parent/'cis_cache_20sdr'; train=CachedCISSet(paths,cache_root,rx_map); loader=DataLoader(train,256,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=4)
    model=ShenNet(len(paths),a.arm=='Shen-RA').to(dev); opt=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=.9)
    a.out.parent.mkdir(parents=True,exist_ok=True); ckpt=a.out.with_suffix('.resume.pt'); start=0
    if a.resume and ckpt.exists():
        state=torch.load(ckpt,map_location=dev,weights_only=False); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); start=state['epoch']; torch.set_rng_state(state['torch_rng'].cpu()); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    writer=SummaryWriter(str(a.out.with_suffix('')/'tensorboard')); history=[]
    for ep in range(start+1,a.epochs+1):
        model.train(); total=correct=0; loss_sum=0.0
        for iq,y,r in loader:
            iq,y,r=iq.to(dev,non_blocking=True),y.to(dev,non_blocking=True),r.to(dev,non_blocking=True); tx,rx=model(iq); loss=nn.functional.cross_entropy(tx,y)+(nn.functional.cross_entropy(rx,r) if rx is not None else 0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); pred=tx.argmax(1); correct+=(pred==y).sum().item(); total+=len(y); loss_sum+=float(loss.detach())
        acc=correct/total; avgloss=loss_sum/max(1,len(loader)); history.append({'epoch':ep,'train_loss':avgloss,'train_acc':acc}); writer.add_scalar('train/loss',avgloss,ep); writer.add_scalar('train/accuracy',acc,ep)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},ckpt)
    writer.close(); payload={'protocol':{'arm':a.arm,'seed':a.seed,'epochs':a.epochs,'batch_size':256,'receivers':len(paths),'validation':'none','checkpoint_selection':'none','blind_opened':False,'optimizer':'SGD','learning_rate':1e-3,'momentum':.9,'cis':'precomputed exact cache'},'parameters':sum(x.numel() for x in model.parameters()),'history':history,'resume_checkpoint':str(ckpt.resolve()),'tensorboard':str((a.out.with_suffix('')/'tensorboard').resolve())}; a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__': main()
