#!/usr/bin/env python3
"""Build the frozen 24-run X2 formal pilot manifest; no signal values opened."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py

FOLDS = ["rtl_2", "rtl_5", "b200_1", "b200_mini_1", "b210_1", "pluto_1"]
BLIND = {"b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2"}

def count(path):
    with h5py.File(path, "r") as f:
        return int(f["data"].shape[0]), sorted(set(int(x) for x in f["label"][:].reshape(-1)))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-root",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    files=sorted(a.source_root.glob("*_train.h5")); names={x.stem.replace("_train","") for x in files}
    if len(files)!=14 or BLIND & names: raise SystemExit("source/blind inventory mismatch")
    inventory={x.stem.replace("_train",""): {"path":str(x.resolve()),"packets":count(x)[0],"labels":count(x)[1]} for x in files}
    runs=[]
    for fold in FOLDS:
        if fold not in names: raise SystemExit(f"missing fold {fold}")
        source=[k for k in sorted(names) if k!=fold]
        for seed in (0,1):
            for model in ("B1","Cprime"):
                runs.append({"fold":fold,"seed":seed,"model":model,"heldout":fold,"source_receivers":source,"lr":1e-3,"weight_decay":5e-4,"optimizer":"AdamW","checkpoint_selection":"source_validation_only","blind_opened":False})
    payload={"protocol":"X2 Formal Pilot","runs":runs,"inventory":inventory,"run_count":len(runs),"total_source_packets_per_fold":sum(v["packets"] for k,v in inventory.items())-inventory[FOLDS[0]]["packets"]}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"runs":len(runs),"files":len(files),"packets_per_file":sorted({v['packets'] for v in inventory.values()}),"blind_opened":False},indent=2))
if __name__=="__main__": main()
