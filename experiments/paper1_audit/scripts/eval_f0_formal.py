#!/usr/bin/env python3
"""Clean held-out evaluation for F0/CT checkpoints."""
import argparse,json
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data import DataLoader
from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier,RFPatchEmbedder
from run_x2_formal import H5Set,views,macro
def main():
 p=argparse.ArgumentParser(); p.add_argument('--ckpt',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--fold',required=True); p.add_argument('--model',choices=['B1','Cprime'],required=True); p.add_argument('--arm',required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); dev=torch.device(a.device); paths=sorted(a.data_root.glob('*_train.h5')); hold=[x for x in paths if x.stem==a.fold+'_train'][0]
 with h5py.File(hold,'r') as f:n=f['data'].shape[0]
 dl=DataLoader(H5Set([hold],{hold.name:list(range(n))}),batch_size=128,shuffle=False,num_workers=2)
 if a.model=='B1': model=MultiViewLateFusionCNN(10).to(dev)
 else:
  e=RFPatchEmbedder(window_size=8192,patch_size=256,dim=64,cnn_stem_dim=32,patch_embed_type='cnn_stem',use_oob=True,oob_fusion_type='cross_attn_oob',use_oob_cross_attention=True,fft_norm='log_zscore',oob_norm='ratio'); model=DeviceClassifier(e,10,dim=64,depth=2,use_chirp_embedding=False).to(dev)
 model.load_state_dict(torch.load(a.ckpt,map_location=dev)['model']); model.eval(); ps=[];ys=[]
 with torch.no_grad():
  for iq,y in dl:
   out=model(views(iq.to(dev)))['logits'] if a.model=='B1' else model(iq.to(dev))['logits']; ps.extend(out.argmax(1).cpu().numpy()); ys.extend(y.numpy())
 payload={'protocol':{'fold':a.fold,'model':a.model,'arm':a.arm,'seed':a.seed,'evaluation':'heldout_clean','blind_opened':False},'accuracy':float(np.mean(np.asarray(ps)==np.asarray(ys))),'macro_f1':macro(np.asarray(ps),np.asarray(ys)),'n_eval':len(ys)}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))
if __name__=='__main__':main()
