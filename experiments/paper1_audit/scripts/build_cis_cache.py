#!/usr/bin/env python3
"""Materialize the exact CIS representation once for final training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
from run_x4c_shen_port import cis

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--batch-size',type=int,default=256); a=p.parse_args(); a.cache_root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for src in sorted(a.data_root.glob('*_train.h5')):
        out=a.cache_root/(src.stem+'.npy'); lab=a.cache_root/(src.stem+'_labels.npy'); meta=a.cache_root/(src.stem+'.json')
        if out.exists() and lab.exists() and meta.exists(): manifest.append(json.loads(meta.read_text())); continue
        with h5py.File(src,'r') as f:
            raw=np.asarray(f['data'],dtype=np.float32); labels=np.asarray(f['label'][0],dtype=np.int64)-31
        t=raw.shape[1]//2; z=raw[:,:t]+1j*raw[:,t:]; z=z/(np.sqrt(np.mean(np.abs(z)**2,axis=1,keepdims=True))+1e-8)
        shape=(len(z),1,52,126); mm=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=shape)
        for start in range(0,len(z),a.batch_size):
            iq=torch.from_numpy(np.stack([z[start:start+a.batch_size].real,z[start:start+a.batch_size].imag],1)).float(); mm[start:start+len(iq)]=cis(iq).numpy()
        mm.flush(); np.save(lab,labels); rec={'source':str(src.resolve()),'cache':str(out.resolve()),'labels':str(lab.resolve()),'shape':list(shape),'dtype':'float32'}; meta.write_text(json.dumps(rec,indent=2)+'\n'); manifest.append(rec); print(src.name,shape)
    (a.cache_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
