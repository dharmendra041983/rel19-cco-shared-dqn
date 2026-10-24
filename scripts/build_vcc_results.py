#!/usr/bin/env python3
"""Build all VCC manuscript numbers, plots and statistical records from raw runs."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'vcc_results';M=ROOT/'manuscript'
M.mkdir(parents=True,exist_ok=True)
D=[json.loads((R/f'seed{i}.json').read_text()) for i in range(5)]
S=[json.loads((R/f'safety_seed{i}.json').read_text()) for i in range(5)]
assert all(d['episodes']==250 and len(d['warm_start'])==2 for d in D)

def values(method,scenario='ring7',metric='coverage'):
 return np.array([d['methods'][method][scenario][metric] for d in D])
def describe(x):
 x=np.asarray(x,float);return {'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'ci95':[float(x.mean()+j*t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5)) for j in [-1,1]]}
def pm(x,scale=100,digits=2):
 q=describe(np.array(x)*scale);return f"${q['mean']:.{digits}f}\\pm{q['sd']:.{digits}f}$"
def ci(x):
 q=describe(x);return f"${q['mean']:.2f}$ pp (paired 95\\% CI $[{q['ci95'][0]:.2f},{q['ci95'][1]:.2f}]$ pp)"
labels={'fixed':'Fixed default','rule':'Rule controller','dqn':'Shared DQN','graph':'Graph DQN','vdn':'VDN','ppo':'Shared PPO'}
scenarios=list(D[0]['methods']['dqn']);scenarios.remove('ring7_unshielded')
stats={'methods':{m:{sc:{k:describe([d['methods'][m][sc][k] for d in D]) for k in D[0]['methods'][m][sc]} for sc in scenarios} for m in labels},'contrasts':{}}
for m in ['graph','vdn','ppo']:
 for sc in ['ring7','ring19','irregular19']:
  stats['contrasts'][f'{m}_minus_dqn_{sc}']=describe(100*(values(m,sc)-values('dqn',sc)))
(M/'vcc_numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{name}}}{{{pm(values(m))}\\%}}' for name,m in [('DqnNominal','dqn'),('GraphNominal','graph'),('VdnNominal','vdn')]))
rows=[]
for m in labels:
 rows.append(labels[m]+' & '+pm(values(m))+' & '+pm(values(m,metric='mean_tput_mbps'),1,3)+' & '+pm(values(m,metric='p5_tput_mbps'),1,3)+r' \\')
(M/'vcc_nominal_table.tex').write_text(r'''\begin{table}[t]
\caption{Held-out nominal seven-cell results. Mean $\pm$ sample standard deviation across five seeds; throughput in Mbps per user. Learned methods use the oracle gate.}
\label{tab:nominal}
\centering\setlength{\tabcolsep}{3pt}\footnotesize
\begin{tabular}{lrrr}\toprule
Controller & Coverage [\%] & Mean rate & 5th percentile\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\end{table}''')
(M/'vcc_nominal_text.tex').write_text('Table~\\ref{tab:nominal} shows that VDN improves nominal coverage over shared DQN by '+ci(100*(values('vdn')-values('dqn')))+'. Graph DQN differs from shared DQN by '+ci(100*(values('graph')-values('dqn')))+'. Thus an explicit team-value objective is beneficial in this seven-cell regime, whereas the learned graph-message layer does not establish an advantage at five seeds. Shared PPO achieves '+pm(values('ppo'))+'\\% coverage under the corrected gradient implementation. All four learned results use the same revised simulator and interaction budget.\n')
(M/'vcc_transfer_text.tex').write_text('Figure~\\ref{fig:transfer} shows coverage on frozen-policy transfer tests. Shared DQN reaches '+pm(values('dqn','ring19'))+'\\% on 19-cell rings and '+pm(values('dqn','irregular19'))+'\\% on irregular 19-cell deployments. Its paired nominal-seven to 19-ring coverage loss is '+ci(100*(values('dqn')-values('dqn','ring19')))+'. VDN reaches '+pm(values('vdn','ring19'))+'\\% on 19-cell rings; its advantage over shared DQN changes from '+ci(100*(values('vdn')-values('dqn')))+' at seven cells to '+ci(100*(values('vdn','ring19')-values('dqn','ring19')))+' at 19 cells. The intervals do not establish a reversed population ranking, but the nominal advantage clearly does not imply uniformly better scale transfer. The changed interference and observations induce distribution shift. The following ablation tests training normalization separately; a positive rescaling of frozen separable utilities cannot itself change greedy actions.\n')
warm={m:{b:np.array([d['warm_start'][m][str(b)]['coverage'] for d in D]) for b in [5,10,25,50]} for m in ['scratch','warm']}
(M/'vcc_warm_text.tex').write_text('Figure~\\ref{fig:warm} compares warm and scratch adaptation. At five online episodes, warm DQN attains '+pm(warm['warm'][5])+'\\% versus '+pm(warm['scratch'][5])+'\\% from scratch, a paired gain of '+ci(100*(warm['warm'][5]-warm['scratch'][5]))+'. At 50 episodes the gain is '+ci(100*(warm['warm'][50]-warm['scratch'][50]))+'. The gain is not monotonic in the online budget, and these checkpoints do not demonstrate a converged-policy advantage.\n')
plt.rcParams.update({'font.size':9,'font.family':'serif','font.serif':['STIXGeneral'],'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.5,2.5))
for m,mark,style,color in [('dqn','o','-','#111111'),('graph','s','--','#4477aa'),('vdn','^','-.','#bb5566'),('ppo','D',':','#228833')]:
 xs=[100*values(m,s) for s in scenarios]
 ax.errorbar(np.arange(8),[x.mean() for x in xs],yerr=[x.std(ddof=1) for x in xs],marker=mark,ls=style,label=labels[m],color=color,capsize=2,lw=1.1,ms=4)
ax.set_xticks(range(8),['Ring\n7','Dense\n7','Sparse\n7','Grid\n7','Irreg.\n7','Ring\n19','Grid\n19','Irreg.\n19'],fontsize=9);ax.set_ylabel('Coverage [%]');ax.legend(ncol=2,fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(M/'vcc_transfer.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.5,2.3))
for name,marker,ls,c in [('warm','o','-','#111111'),('scratch','s','--','#4477aa')]:
 xs=[100*warm[name][b] for b in [5,10,25,50]]
 ax.errorbar([5,10,25,50],[x.mean() for x in xs],yerr=[x.std(ddof=1) for x in xs],marker=marker,ls=ls,label=name.capitalize(),color=c,capsize=3)
ax.set_xlabel('Online training episodes');ax.set_ylabel('Coverage [%]');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(M/'vcc_warm.pdf');plt.close(fig)
rows=[];safety={}
for sc,label in [('ring7','Ring 7'),('irregular7','Irregular 7'),('ring19','Ring 19')]:
 safety[sc]={}
 for mode,lab in [('none','None'),('oracle','Oracle'),('surrogate','Learned')]:
  x={k:np.array([d['scenarios'][sc][mode][k] for d in S]) for k in ['coverage','violation_pct','reject_pct']}
  safety[sc][mode]={k:describe(v) for k,v in x.items()}
  rows.append(label+' & '+lab+' & '+pm(x['coverage'])+' & '+pm(x['violation_pct'],1)+' & '+pm(x['reject_pct'],1)+r' \\')
(M/'vcc_safety_table.tex').write_text(r'''\begin{table*}[t]
\caption{Frozen-DQN gate audit: applied coverage-floor violations and rejected actions, both as percentages of epochs. Each entry is mean $\pm$ standard deviation over five seeds; Learned uses mixed-behavior q95 calibration.}
\label{tab:gate}\centering\footnotesize
\begin{tabular}{llrrr}\toprule
Test condition & Gate & Coverage [\%] & Violations [\%] & Rejections [\%]\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{table*}''')
text='Table~\\ref{tab:gate} compares the same frozen DQN under three gates. '
for sc,lab in [('ring7','nominal seven-cell rings'),('irregular7','irregular seven-cell layouts'),('ring19','19-cell rings')]:
 x=safety[sc]['surrogate'];n=safety[sc]['none']
 text+=f"On {lab}, the learned gate has {x['violation_pct']['mean']:.2f}\\% applied violations and rejects {x['reject_pct']['mean']:.2f}\\% of commands; no screening gives {n['violation_pct']['mean']:.2f}\\% violations. "
text+='The exact gate has zero applied violations, as required by its acceptance rule. Calibration margins range from '+f"${min(s['margin_pp'] for s in S):.2f}$ to ${max(s['margin_pp'] for s in S):.2f}$ pp across seeds. These results quantify the cost of replacing counterfactual information rather than establish safety outside the tested rollouts.\n"
(M/'vcc_safety_text.tex').write_text(text)
stats['safety']=safety;stats['warm_start']={m:{b:describe(v) for b,v in bs.items()} for m,bs in warm.items()}
(R/'aggregated.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats['contrasts'],indent=2));print(text)
