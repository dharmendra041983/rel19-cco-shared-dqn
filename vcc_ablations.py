#!/usr/bin/env python3
"""Prespecified VDN training-normalization and frozen-policy invariance audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,json
import numpy as np
import vcc_experiments as v

class MeanVDN(v.Agent):
    def learn(self):
        if len(self.buffer)<80:return
        ix=self.rng.integers(0,len(self.buffer),64)
        s,a,u,r,t=map(np.array,zip(*(self.buffer[i] for i in ix)))
        q,cache=self.net.forward(s,a);qt,_=self.target.forward(t,a)
        b,n=u.shape;chosen=np.take_along_axis(q,u[...,None],-1)[...,0]
        td=chosen.mean(1)-(r+.8*qt.max(-1).mean(1))
        deriv=np.repeat((np.clip(td,-1,1)/(b*n))[:,None],n,axis=1)
        dy=np.zeros_like(q);np.put_along_axis(dy,u[...,None],deriv[...,None],-1)
        self.opt.step(self.net.params(),self.net.backward(cache,dy));self.updates+=1
        if self.updates%400==0:self.target=copy.deepcopy(self.net)

def frozen_invariance(agent,cfg):
    # Positive scalar team-value normalization leaves separable argmax intact.
    tested=disagreements=0
    for sc in v.SCENARIOS:
        c=dict(cfg,**v.SCENARIOS[sc])
        for ts in (10,11,12,13):
            env=v.Env(ts,c);v.base._reset_eval_env(env,cfg['seed'],ts,0);a=v.adjacency(env)
            for _ in range(40):
                q,_=agent.net.forward(env.features()[None],a[None]);q=q[0]
                raw=q.argmax(1);normalized=(q/env.n).argmax(1)
                disagreements+=int(np.sum(raw!=normalized));tested+=env.n
                env.step(raw)
    return {'cell_decisions':tested,'normalization_disagreements':disagreements}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();seed=args.seed
    cfg=dict(v.base.CFG,seed=seed)
    # Same initial utility weights, training seeds, replay samples, optimizer LR,
    # interactions and architecture. Mixer changes targets and gradient scaling.
    agent=MeanVDN(100+seed,'vdn');v.train_value(agent,cfg,250)
    result={'seed':seed,'episodes':250,'definition':'mean of utility values; unchanged scalar reward and optimizer hyperparameters','conditions':{sc:v.evaluate(agent,cfg,sc) for sc in v.SCENARIOS}}
    agent.save(v.OUT/f'vdn_mean_seed{seed}.npz')
    raw=v.Agent(100+seed,'vdn');w=np.load(v.OUT/f'vdn_seed{seed}.npz')
    for i,p in enumerate(raw.net.params()):p[:]=w[f'p{i}']
    result['frozen_sum_invariance']=frozen_invariance(raw,cfg)
    (v.OUT/f'normalization_seed{seed}.json').write_text(json.dumps(result,indent=2));print('normalization',seed,'COMPLETE',flush=True)
if __name__=='__main__':main()
