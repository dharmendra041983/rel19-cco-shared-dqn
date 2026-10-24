#!/usr/bin/env python3
"""Frozen predictor/policy margin sweep with no held-out tuning.
Also recalibrate on independent oracle-gated DQN rollouts on training layouts.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import vcc_experiments as v
import vcc_safety as g

class FastForest:
    """Use sklearn trees' compiled predict calls without per-tree estimator setup."""
    def __init__(self,model):self.model=model
    def predict(self,x):
        x=np.asarray(x,dtype=np.float32,order='C')
        return sum(t.tree_.predict(x).reshape(len(x),-1)[:,0] for t in self.model.estimators_)/len(self.model.estimators_)

def policy_calibration(seed,agent,model):
    # Frozen DQN actions, training geometry IDs only, independent user RNG.
    cfg=dict(v.base.CFG,seed=seed);res=[]
    for ep in range(25):
        env=v.Env(ep%6,cfg);env.rng=np.random.default_rng(9000000+100000*seed+ep);env.reset(True);a=v.adjacency(env)
        for _ in range(40):
            u=agent.act(env.features(),a);env._churn();env._compute();ref=env.kpis()['coverage'];x=g.features(env,u)
            old=(env.tilt.copy(),env.power.copy());env.apply(u);env._compute();delta=env.kpis()['coverage']-ref
            res.append(float(model.predict(x[None])[0])-delta)
            if delta<-.02-1e-12:env.tilt,env.power=old;env._compute()
    return np.array(res)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();seed=args.seed
    x,y=g.data(seed,100,7000000);xc,yc=g.data(seed,25,8000000)
    forest=RandomForestRegressor(n_estimators=64,max_depth=12,min_samples_leaf=20,random_state=seed,n_jobs=1).fit(x,y)
    model=FastForest(forest);assert np.allclose(model.predict(xc[:15]),forest.predict(xc[:15]),atol=1e-12)
    joblib.dump(forest,v.OUT/f'gate_seed{seed}.joblib')
    agent=v.Agent(100+seed,'dqn');weights=np.load(v.OUT/f'dqn_seed{seed}.npz')
    for i,p in enumerate(agent.net.params()):p[:]=weights[f'p{i}']
    residual=model.predict(xc)-yc;pr=policy_calibration(seed,agent,model)
    margins={'zero':0.,'fixed_1pp':.01,'fixed_2pp':.02,
      **{f'mixed_q{int(q*100)}':max(0,float(np.quantile(residual,q,method='higher'))) for q in [.8,.9,.95,.99]},
      'policy_q95':max(0,float(np.quantile(pr,.95,method='higher')))}
    result={'seed':seed,'margins_pp':{k:100*m for k,m in margins.items()},'scenarios':{}}
    for sc in ['ring7','irregular7','ring19']:
        result['scenarios'][sc]={name:g.audit(seed,model,m,agent,sc,'surrogate') for name,m in margins.items()}
        print('margin',seed,sc,'complete',flush=True)
    (v.OUT/f'margin_seed{seed}.json').write_text(json.dumps(result,indent=2));print('margin',seed,'COMPLETE',flush=True)
if __name__=='__main__':main()
