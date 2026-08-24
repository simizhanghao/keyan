#!/usr/bin/env python3
"""Build the immutable pre-blind Paper-B checkpoint and environment inventory."""
from __future__ import annotations
import hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np, torch
try:
 import h5py
 h5py_version = h5py.__version__
except Exception:
 h5py_version = 'not-installed (snapshot does not require HDF5 I/O)'

ROOT=Path(__file__).resolve().parents[1]
# ``new_phase`` is the nested git repository; the script lives three levels below it.
REPO=ROOT.parents[2]
RESULTS=ROOT/'results/final_training'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def cmd(*a): return subprocess.check_output(a,text=True,stderr=subprocess.DEVNULL).strip()
def add(rows,model,seed,json_path):
 x=json.loads(json_path.read_text()); proto=x['protocol']; p=Path(x.get('checkpoint') or x['resume_checkpoint']); state=torch.load(p,map_location='cpu',weights_only=False)
 rows.append({'model':model,'seed':seed,'epoch':int(state['epoch']),'parameters':int(x['parameters']),'checkpoint':str(p.resolve()),'checkpoint_sha256':sha(p),'metrics_json':str(json_path.resolve()),'metrics_sha256':sha(json_path),'batch_size':int(proto['batch_size']),'blind_opened':bool(proto['blind_opened'])})
def main():
 rows=[]
 for s in range(4): add(rows,'Shen-CIS',s,RESULTS/f'shen_clean/Shen-CIS/seed_{s}.json')
 add(rows,'Shen-CIS',4,RESULTS/'shen/Shen-CIS/seed_4.json')
 for s in range(5): add(rows,'Shen-RA',s,RESULTS/f'shen/Shen-RA/seed_{s}.json')
 for model,folder in [('B1-OOB','B1-OOB'),("C'-OOB",'Cprime-OOB'),("C'-TrueIB",'Cprime-TrueIB')]:
  for s in range(5): add(rows,model,s,RESULTS/f'short_clean/{folder}/seed_{s}.json')
 assert len(rows)==25 and all(not r['blind_opened'] and r['batch_size']==64 for r in rows)
 code=[]
 for name in ['run_final_shen.py','run_final_short.py','build_cis_cache.py','run_x3_scale.py','run_x3_shuffle.py','run_x3_occlusion.py','build_x6_neutral.py','run_x6_confirmatory.py']:
  p=ROOT/'scripts'/name; code.append({'path':str(p.relative_to(REPO)),'sha256':sha(p)})
 artifacts=[]
 for name in ['FINAL_DATA_MANIFEST.md','THREE_PAPER_BOUNDARY_LOCK.md','PAPER_C_PREREG_LOCK.md','X6_PROTOCOL_LOCK.md','results/final_training/x6_neutral/development_neutral.npy','results/final_training/x6_neutral/development_neutral.json']:
  p=ROOT/name; artifacts.append({'path':str(p.relative_to(REPO)),'sha256':sha(p)})
 env={'python':platform.python_version(),'pytorch':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy':np.__version__,'h5py':h5py_version,'platform':platform.platform(),'git_parent':cmd('git','-C',str(REPO),'rev-parse','HEAD')}
 try: env['gpu']=cmd('nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader').splitlines()[0]
 except Exception: env['gpu']='unavailable during snapshot build'
 payload={'status':'FROZEN_BEFORE_X6','checkpoint_count':25,'development_receivers':14,'development_packets':112000,'official_blind_receivers':6,'official_blind_status':'SEALED','blind_archive':{'path':'/data1/hcc/llm4RF/data0820/multiple_receiver_test.zip','sha256':'e8f4cdc32cbbb7e6cfa410375e7412e204a2c412996d2a22a55ab0c8b9ff79f1','signal_values_opened':False},'checkpoints':rows,'code':code,'artifacts':artifacts,'environment':env,'excluded':['results/final_training/shen/Shen-CIS/seed_0..3 (mixed batch trajectory)','results/final_training/short/* (no saved checkpoints)']}
 out=ROOT/'FINAL_EXPERIMENT_SNAPSHOT.json'; out.write_text(json.dumps(payload,indent=2)+'\n')
 md=['# Final Experiment Snapshot','','Status: **FROZEN BEFORE X6**. Official blind receivers were sealed when this snapshot was generated.','',f"- Canonical checkpoints: `{len(rows)}`",'- Development receivers/packets: `14 / 112000`','- Official blind receivers: `6 (SEALED)`',f"- Git parent: `{env['git_parent']}`",'', '## Canonical checkpoints','', '| Model | Seed | Epoch | Params | SHA256 |','|---|---:|---:|---:|---|']
 md += [f"| {r['model']} | {r['seed']} | {r['epoch']} | {r['parameters']} | `{r['checkpoint_sha256']}` |" for r in rows]
 md += ['', '## Environment','']+[f"- `{k}`: `{v}`" for k,v in env.items()]+['','## Code hashes','']+[f"- `{r['path']}`: `{r['sha256']}`" for r in code]+['','## Frozen artifact hashes','']+[f"- `{r['path']}`: `{r['sha256']}`" for r in artifacts]+['', '## Blind archive', '', f"- SHA256: `{payload['blind_archive']['sha256']}`", '- Signal values opened: `false`', '', '## Excluded artifacts','','- Mixed-batch Shen-CIS seeds 0-3 under `results/final_training/shen/`.','- No-checkpoint short runs under `results/final_training/short/`.','','After first blind access, retraining, checkpoint reselection, hyperparameter tuning, preprocessing changes, intervention changes, and model additions are prohibited.','']
 (ROOT/'FINAL_EXPERIMENT_SNAPSHOT.md').write_text('\n'.join(md))
 print(out, len(rows))
if __name__=='__main__': main()
