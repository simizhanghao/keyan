#!/usr/bin/env python3
"""Estimate source-derived neutral OOB magnitude vectors, excluding each fold."""
from pathlib import Path
import argparse, h5py, numpy as np
FOLDS=('rtl_2','rtl_5','b200_1','b200_mini_1','b210_1','pluto_1')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--out-root',type=Path,required=True); a=p.parse_args(); paths=sorted(a.data_root.glob('*_train.h5')); mask=np.abs(np.fft.fftshift(np.fft.fftfreq(8192,d=1e-6)))>62500
 for fold in FOLDS:
  vals=[]
  for path in paths:
   if path.stem==fold+'_train': continue
   with h5py.File(path,'r') as f:
    n=min(256,f['data'].shape[0]); raw=np.asarray(f['data'][:n],dtype=np.float32); t=raw.shape[1]//2; z=raw[:,:t]+1j*raw[:,t:]; z=z/(np.sqrt(np.mean(np.abs(z)**2,axis=1,keepdims=True))+1e-8); vals.append(np.abs(np.fft.fftshift(np.fft.fft(z,axis=1),axes=1))[:,mask].mean(axis=0))
  a.out_root.mkdir(parents=True,exist_ok=True); np.save(a.out_root/f'{fold}.npy',np.mean(vals,axis=0).astype(np.float32)); print(fold, len(vals))
if __name__=='__main__':main()
