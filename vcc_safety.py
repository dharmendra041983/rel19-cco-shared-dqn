#!/usr/bin/env python3
"""Frozen-DQN audit of exact and approximate action screening.
The learned gate uses only aggregate telemetry/configuration and proposed
commands. Simulator counterfactuals are used for labels and evaluation only.
No safety probability guarantee is asserted under topology shift.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import vcc_experiments as v

def features(env,u):
 s=env.features();commands=np.eye(5)[u]
 return np.concatenate([s.mean(0),s.std(0),commands.mean(0),(commands[:,:,None]*s[:,None,:]).mean(0).ravel()])

def data(seed,episodes,offset):
 cfg=dict(v.base.CFG,seed=seed);rng=np.random.default_rng(seed+offset);xx=[];yy=[]
 for ep in range(episodes):
  env=v.Env(ep%6,cfg);env.rng=np.random.default_rng(offset+seed*100000+ep);env.reset(True)
  for t in range(40):
   env._churn();env._compute();ref=env.kpis()['coverage']
   u=v.base.rule_policy(env) if t%2 else rng.integers(0,5,env.n)
   x=features(env,u);old=(env.tilt.copy(),env.power.copy());env.apply(u);env._compute()
   delta=env.kpis()['coverage']-ref;xx.append(x);yy.append(delta)
   if delta<-.02:env.tilt,env.power=old;env._compute()
 return np.array(xx),np.array(yy)

def audit(seed,model,margin,agent,sc,mode):
 cfg=dict(v.base.CFG,seed=seed,**v.SCENARIOS[sc]);count=reject=viol=0;cov=[];errors=[];drops=[]
 for ts in (10,11,12,13):
  for init in range(3):
   env=v.Env(ts,cfg);v.base._reset_eval_env(env,seed,ts,init);a=v.adjacency(env)
   for _ in range(40):
    u=agent.act(env.features(),a)
    env._churn();env._compute();ref=env.kpis()['coverage'];x=features(env,u)
    predicted=float(model.predict(x[None])[0]) if mode=='surrogate' else 0.
    old=(env.tilt.copy(),env.power.copy());env.apply(u);env._compute();delta=env.kpis()['coverage']-ref
    # 'surrogate' branch does not use delta for its acceptance decision.
    reject_action=(delta<-.02-1e-12) if mode=='oracle' else (predicted-margin<-.02 if mode=='surrogate' else False)
    if reject_action:env.tilt,env.power=old;env._compute();reject+=1
    applied=env.kpis()['coverage']-ref;viol+=applied<-.02-1e-12;drops.append(max(0,-applied));count+=1
    cov.append(env.kpis()['coverage'])
    if mode=='surrogate':errors.append(abs(predicted-delta))
 return {'coverage':float(np.mean(cov)),'violation_pct':100*viol/count,'reject_pct':100*reject/count,'max_drop_pp':100*max(drops),'prediction_mae_pp':100*float(np.mean(errors)) if errors else None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();seed=args.seed
 x,y=data(seed,100,7000000);xc,yc=data(seed,25,8000000)
 model=RandomForestRegressor(n_estimators=64,max_depth=12,min_samples_leaf=20,random_state=seed,n_jobs=1).fit(x,y)
 residual=model.predict(xc)-yc;margin=max(0,float(np.quantile(residual,.95,method='higher')))
 agent=v.Agent(100+seed,'dqn');weights=np.load(v.OUT/f'dqn_seed{seed}.npz')
 for i,p in enumerate(agent.net.params()):p[:]=weights[f'p{i}']
 result={'seed':seed,'train_samples':len(y),'calibration_samples':len(yc),'margin_pp':100*margin,'scenarios':{}}
 for sc in ['ring7','irregular7','ring19']:
  result['scenarios'][sc]={mode:audit(seed,model,margin,agent,sc,mode) for mode in ['none','oracle','surrogate']}
 (v.OUT/f'safety_seed{seed}.json').write_text(json.dumps(result,indent=2));print('safety',seed,'complete',flush=True)
if __name__=='__main__':main()
