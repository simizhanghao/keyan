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
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})

def save(fig, name):
    fig.tight_layout(pad=.8)
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.tiff'), dpi=600, bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7.16,2.55)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,3.2)
boxes=[(.15,1.18,1.45,.92,'Transmitter\nimpairments','#dbeafe','///'),(2.05,1.18,1.45,.92,'RF capture\nI/Q','#e0f2fe','...'),(4.18,1.76,1.35,.68,'In-band\nevidence','#dcfce7','///'),(4.18,.84,1.35,.68,'OOB\nevidence','#fee2e2','...'),(6.18,1.18,1.65,.92,'Receiver front-end\nand acquisition','#fef3c7','///'),(8.48,1.18,1.3,.92,'Learned\nrepresentation','#ede9fe','...')]
for x,y,w,h,t,c,hatch in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=c,edgecolor='#1f2937',lw=.8,hatch=hatch)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9,color='#111827')
for a,b in [((1.6,1.64),(2.05,1.64)),((3.5,1.64),(4.18,2.10)),((3.5,1.64),(4.18,1.18)),((5.53,2.10),(6.18,1.64)),((5.53,1.18),(6.18,1.64)),((7.83,1.64),(8.48,1.64))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=.9,color='#374151'))
ax.text(7.55,2.82,'receiver-sensitive OOB dependence',ha='center',va='center',fontsize=9,color='#7c2d12',weight='bold')
ax.add_patch(FancyArrowPatch((7.55,2.68),(7.55,2.48),arrowstyle='-|>',mutation_scale=9,lw=.9,ls='--',color='#9a3412'))
ax.text(5.1,.28,'Study interventions: OOB scaling  |  same-receiver shuffle  |  neutral replacement  |  left/right perturbation',ha='center',fontsize=7.5,color='#374151')
save(fig,'fig1_concept')

groups=['Device','RX type','RX instance','Day']; vals=[.393,.111,.083,.015]; order=np.argsort(vals)
fig,ax=plt.subplots(figsize=(3.5,2.45)); bars=ax.barh(np.arange(len(groups)),np.array(vals)[order],color=['#b0b0b0','#72b7b2','#f2cf5b','#4c78a8'],edgecolor='black',linewidth=.45,hatch=['///','...','\\\\\\\\',''])
ax.set_yticks(np.arange(len(groups)),np.array(groups)[order]); ax.set_xlabel('SD of group means (dB)'); ax.set_xlim(0,.44); ax.grid(axis='x',alpha=.22); ax.invert_yaxis()
for b,v in zip(bars,np.array(vals)[order]): ax.text(v+.008,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=8)
save(fig,'fig2_signal_audit')

rx=['rtl$_2$','rtl$_5$','b200$_1$','b200-mini$_1$','b210$_1$','pluto$_1$']; x=np.arange(6)
fig,axs=plt.subplots(1,2,figsize=(7,2.55),sharey=True)
for ax,title,b1,cp in [(axs[0],'OOB scale',[46.79,55.63,34.09,47.78,49.38,37.11],[16.54,38.68,59.57,64.87,53.62,75.42]),(axs[1],'Same-RX cross-device shuffle',[45.89,57.21,33.38,50.04,48.89,40.28],[17.59,41.74,64.99,71.23,56.81,81.91])]:
    ax.plot(x,b1,'o-',label='LateFusion-OOB',color='#2563eb',markerfacecolor='white',markeredgewidth=1.2); ax.plot(x,cp,'s--',label="CrossAttn-OOB",color='#dc2626',markerfacecolor='white',markeredgewidth=1.2); ax.set_title(title); ax.set_xticks(x,rx,rotation=30); [t.set_rotation_mode('anchor') for t in ax.get_xticklabels()]; ax.set_ylabel('Accuracy drop (pp)'); ax.grid(alpha=.2); ax.legend(frameon=False)
save(fig,'fig3_development_interventions')

labels=['LateFusion\nOOB','LateFusion\nTrueIB','CrossAttn\nOOB',"CrossAttn\nTrueIB",'Shen-\nCIS','Shen-\nRA']; vals=[56.64,11.97,60.90,70.91,94.101,93.847]
fig,ax=plt.subplots(figsize=(4.8,2.7)); bars=ax.bar(labels,vals,color=['#2563eb','#93c5fd','#dc2626','#fca5a5','#059669','#34d399'],edgecolor='black',linewidth=.35,hatch=['///','...','///','...','///','...']); ax.set_ylabel('Held-out accuracy (%)'); ax.set_ylim(0,105); ax.set_title('Matched controls on development folds'); ax.grid(axis='y',alpha=.2)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=7)
save(fig,'fig4_controls')

fig,axs=plt.subplots(1,2,figsize=(7,2.55))
labels=['Clean\ncontinuation (CT)','RSA/F0']; drops=[53.96,2.92]
bars=axs[0].bar(labels,drops,color=['#64748b','#f59e0b'],edgecolor='black',linewidth=.4,hatch=['///','...'])
axs[0].set_ylabel('Full-scale accuracy drop (pp)'); axs[0].set_ylim(0,65); axs[0].set_title('(a) Controlled scale response'); axs[0].grid(axis='y',alpha=.2)
for b,v in zip(bars,drops): axs[0].text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.2f}',ha='center',fontsize=8)
clean=[64.25625,60.85156]; sd=[14.1624,16.3994]
bars=axs[1].bar(labels,clean,yerr=sd,capsize=3,color=['#64748b','#f59e0b'],edgecolor='black',linewidth=.4,hatch=['///','...'])
axs[1].set_ylabel('Clean held-out accuracy (%)'); axs[1].set_ylim(0,100); axs[1].set_title('(b) Real receiver transfer'); axs[1].grid(axis='y',alpha=.2)
for b,v in zip(bars,clean): axs[1].text(b.get_x()+b.get_width()/2,v+3,f'{v:.2f}',ha='center',fontsize=8)
fig.suptitle('Controlled scale invariance does not imply receiver generalization',fontsize=9)
save(fig,'fig5_mitigation_boundary')

summary=json.loads((AUDIT/'results/x6_blind/X6_SUMMARY.json').read_text()); rows=summary['rows']
receivers=['b200_2','b200_mini_2','b210_2','n210_2','n210_3','pluto_2']; x=np.arange(6)
display={'B1-OOB':'LateFusion-OOB',"C'-OOB":'CrossAttn-OOB'}; colors={'B1-OOB':'#2563eb',"C'-OOB":'#dc2626'}
def acc(model,receiver,condition):
    vals=[r['accuracy'] for r in rows if r['model']==model and r['receiver']==receiver and r['condition']==condition]
    return float(np.mean(vals))*100
fig,axs=plt.subplots(2,3,figsize=(7,4.8),sharex='col')
for i,(model,col) in enumerate(zip(['B1-OOB',"C'-OOB"],['#2563eb','#dc2626'])):
    utility=[]; worst=[]; left=[]; right=[]
    for receiver in receivers:
        clean=acc(model,receiver,'clean')
        utility.append((clean-acc(model,receiver,'shuffle'),clean-acc(model,receiver,'neutral')))
        scales=[acc(model,receiver,'scale_'+s) for s in ['0.5','0.70710678','1.41421356','2.0']]
        worst.append(clean-min(scales))
        left.append(clean-acc(model,receiver,'left_scale_0.5')); right.append(clean-acc(model,receiver,'right_scale_0.5'))
    ax=axs[i,0]; w=.34; ax.bar(x-w/2,[v[0] for v in utility],w,color=col,edgecolor='black',hatch='///',label='Shuffle'); ax.bar(x+w/2,[v[1] for v in utility],w,color=col,edgecolor='black',hatch='...',label='Neutral'); ax.set_title(f'{display[model]}: OOB utility'); ax.legend(frameon=False,fontsize=6); ax.set_ylabel('Accuracy drop (pp)')
    ax=axs[i,1]; ax.bar(x,worst,color=col,edgecolor='black',hatch='//'); ax.set_title('Worst-scale sensitivity')
    ax=axs[i,2]; w=.34; ax.bar(x-w/2,left,w,color=col,edgecolor='black',hatch='\\\\',label='Left'); ax.bar(x+w/2,right,w,color=col,edgecolor='black',hatch='..',label='Right'); ax.set_title('Frequency asymmetry'); ax.legend(frameon=False,fontsize=6)
    for j in range(3): axs[i,j].grid(axis='y',alpha=.2)
for ax in axs[-1,:]:
    ax.set_xticks(x,['b200$_2$','b200-mini$_2$','b210$_2$','n210$_2$','n210$_3$','pluto$_2$'],rotation=35)
    [t.set_rotation_mode('anchor') for t in ax.get_xticklabels()]
fig.suptitle('Protocol-frozen blind confirmation across six receiver domains',fontsize=9)
save(fig,'fig6_blind_confirmation')
