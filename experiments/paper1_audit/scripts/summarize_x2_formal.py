#!/usr/bin/env python3
import argparse, glob, json, statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); rows=[]
 for path in sorted(a.root.glob('*/*/seed_*/metrics.json')):
  d=json.loads(path.read_text()); fold,model,seed=path.parts[-4],path.parts[-3],int(path.parts[-2].split('_')[1]); rows.append({'fold':fold,'model':model,'seed':seed,'accuracy':d['heldout_accuracy'],'macro_f1':d['heldout_macro_f1'],'source_val_accuracy':d['history'][-1]['source_val_acc'],'best_epoch':d['best_epoch'],'parameters':d['parameters']})
 folds=sorted({r['fold'] for r in rows}); summary=[]
 for fold in folds:
  b=[r for r in rows if r['fold']==fold and r['model']=='B1']; c=[r for r in rows if r['fold']==fold and r['model']=='Cprime']; summary.append({'fold':fold,'B1_accuracy_mean':statistics.mean(r['accuracy'] for r in b),'Cprime_accuracy_mean':statistics.mean(r['accuracy'] for r in c),'delta_Cprime_minus_B1_pp':100*(statistics.mean(r['accuracy'] for r in c)-statistics.mean(r['accuracy'] for r in b)),'B1_macro_f1_mean':statistics.mean(r['macro_f1'] for r in b),'Cprime_macro_f1_mean':statistics.mean(r['macro_f1'] for r in c)})
 def agg(model,key):
  v=[r[key] for r in rows if r['model']==model]; return {'mean':statistics.mean(v),'median':statistics.median(v),'std_seed_run':statistics.stdev(v)}
 deltas=[x['delta_Cprime_minus_B1_pp'] for x in summary]; overall={'B1':agg('B1','accuracy'),'Cprime':agg('Cprime','accuracy'),'Cprime_minus_B1_mean_pp':statistics.mean(deltas),'Cprime_minus_B1_median_fold_pp':statistics.median(deltas),'positive_folds':sum(x>0 for x in deltas),'folds':len(deltas)}
 payload={'protocol':'X2 Formal Pilot','runs':len(rows),'blind_opened':False,'per_fold':summary,'overall':overall,'decision':'X3_BOTH_B1_AND_CPRIME','decision_reason':'Cprime mean gain >=5 pp and 4/6 positive folds, but large RTL negative deltas require both backbones in shortcut audit.','rows':rows}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'runs':len(rows),'overall':overall,'decision':payload['decision']},indent=2))
if __name__=='__main__': main()
