#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent.parent / 'new_phase/experiments/paper1_audit'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42})

def save(fig, name):
    fig.tight_layout(pad=.8)
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7,2.3)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,3)
boxes=[(.2,1,1.25,1,'Transmitter\nimpairments','#dbeafe'),(2,1,1.35,1,'RF capture\nI/Q window','#e0f2fe'),(4,1.48,1.4,.72,'In-band','#dcfce7'),(4,.55,1.4,.72,'OOB spectrum','#fee2e2'),(6.2,1,1.55,1,'Receiver front-end\nresponse','#fef3c7'),(8.55,1,1.2,1,'Classifier\ndecision','#ede9fe')]
for x,y,w,h,t,c in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=c,edgecolor='#334155',lw=.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center')
for a,b in [((1.45,1.5),(2,1.5)),((3.35,1.5),(4,1.84)),((3.35,1.5),(4,.91)),((5.4,1.84),(6.2,1.5)),((5.4,.91),(6.2,1.5)),((7.75,1.5),(8.55,1.5))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=10,lw=1,color='#475569'))
ax.text(6.98,2.45,'receiver-sensitive\nOOB dependence',ha='center',color='#b45309',weight='bold')
ax.text(6.98,.25,'scale | shuffle | neutral | left/right interventions',ha='center',color='#475569')
save(fig,'fig1_concept')

groups=['Device','Receiver type','Receiver instance','Day']; vals=[.393,.111,.083,.015]
fig,ax=plt.subplots(figsize=(3.4,2.4)); ax.bar(groups,vals,color=['#2563eb','#f59e0b','#10b981','#94a3b8']); ax.set_ylabel('SD of group means (dB)'); ax.set_title('Signal-level OOB/IB variation'); ax.tick_params(axis='x',labelrotation=25); ax.grid(axis='y',alpha=.2); save(fig,'fig2_signal_audit')

rx=['rtl$_2$','rtl$_5$','b200$_1$','b200-mini$_1$','b210$_1$','pluto$_1$']; x=np.arange(6)
fig,axs=plt.subplots(1,2,figsize=(7,2.55),sharey=True)
for ax,title,b1,cp in [(axs[0],'Scale sensitivity',[46.79,55.63,34.09,47.78,49.38,37.11],[16.54,38.68,59.57,64.87,53.62,75.42]),(axs[1],'Same-RX cross-device shuffle',[45.89,57.21,33.38,50.04,48.89,40.28],[17.59,41.74,64.99,71.23,56.81,81.91])]:
    ax.plot(x,b1,'o-',label='B1-OOB',color='#2563eb'); ax.plot(x,cp,'s-',label="C'-OOB",color='#dc2626'); ax.set_title(title); ax.set_xticks(x,rx,rotation=30); ax.set_ylabel('Accuracy drop (pp)'); ax.grid(alpha=.2); ax.legend(frameon=False)
save(fig,'fig3_development_interventions')

labels=['B1\nOOB-aware','B1\nTrueIB',"C'\nOOB-aware","C'\nTrueIB",'Shen-CIS','Shen-RA']; vals=[56.64,11.97,60.90,70.91,94.101,93.847]
fig,ax=plt.subplots(figsize=(4.8,2.7)); bars=ax.bar(labels,vals,color=['#2563eb','#93c5fd','#dc2626','#fca5a5','#059669','#34d399']); ax.set_ylabel('Held-out accuracy (%)'); ax.set_ylim(0,105); ax.set_title('Matched controls on development folds'); ax.grid(axis='y',alpha=.2)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=7)
save(fig,'fig4_controls')

labels=['CT\nclean','RSA\nclean','RSA\nafter scale probe']; vals=[53.96,53.96,2.92]
fig,ax=plt.subplots(figsize=(3.8,2.5)); bars=ax.bar(labels,vals,color=['#64748b','#f59e0b','#ef4444']); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,65); ax.set_title('Controlled invariance is not transfer'); ax.grid(axis='y',alpha=.2)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.2f}',ha='center',fontsize=8)
save(fig,'fig5_mitigation_boundary')

summary=json.loads((AUDIT/'results/x6_blind/X6_SUMMARY.json').read_text()); eff=summary['receiver_effects']; fig,axs=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
for ax,model,col in zip(axs,['B1-OOB',"C'-OOB"],['#2563eb','#dc2626']):
    rows=[r for r in eff if r['model']==model]; xx=np.arange(6); ax.bar(xx,[r['disruption_drop']*100 for r in rows],color=col,alpha=.85,label='shuffle + neutral'); ax.plot(xx,[r['worst_scale_drop']*100 for r in rows],'ko--',ms=4,label='worst scale'); ax.set_title(model); ax.set_xticks(xx,['b200$_2$','b200-mini$_2$','b210$_2$','n210$_2$','n210$_3$','pluto$_2$'],rotation=35); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,loc='upper left')
axs[0].set_ylabel('Accuracy drop (pp)'); fig.suptitle('Official blind confirmation: 6/6 receivers'); save(fig,'fig6_blind_confirmation')
