#!/usr/bin/env python3
"""VCC revision: matched NumPy controllers, topology shifts and oracle-free audit.
Archived IPCCC outputs are never overwritten. See VCC_README.md for semantics.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import cco_experiments_corrected as base
from ppo_safety_scale_experiments import PPO

OUT = Path(__file__).parent / 'vcc_results'
OUT.mkdir(exist_ok=True)
ORIGINAL_TOPOLOGY = base.make_topology

def topology(seed, cfg):
    family = cfg.get('layout', 'ring')
    if family == 'ring':
        return ORIGINAL_TOPOLOGY(seed, cfg)
    rng = np.random.default_rng(70000 + seed)
    n, d = cfg['n_sites'], cfg['isd']
    if family == 'grid':
        side = int(np.ceil(np.sqrt(n)))
        p = np.array([(i % side, i // side) for i in range(n)], float) * d
        p -= p.mean(0)
        p += rng.normal(0, .08*d, p.shape)
    elif family == 'irregular':
        pts = []
        radius = d*np.sqrt(n/3)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 2)
            if not pts or np.min(np.linalg.norm(np.array(pts)-p, axis=1)) > .45*d:
                pts.append(p)
        p = np.array(pts); p -= p.mean(0)
    else:
        raise ValueError(family)
    return p
base.make_topology = topology

class Env(base.Env):
    def _churn(self):
        # Replace unique users and their hotspot labels consistently.
        m = self.cfg['n_ues']; k = min(m, max(1, int(m*self.cfg['churn_frac'])))
        idx = self.rng.choice(m, k, replace=False)
        pos, shad, hot = self._drop_users(k)
        self.ue[idx] = pos; self.shadow[:, idx] = shad; self.is_hot[idx] = hot
        self.ue[self.is_hot] += self.rng.normal(0, self.cfg['jitter_m'], (self.is_hot.sum(),2))
    def reward(self):
        k = self.kpis()
        changes = np.abs(self.tilt-self.prev_tilt) + np.abs(self.power-self.prev_power)
        # Equal to original objective at seven cells; stable scale elsewhere.
        r = 10*k['coverage'] + .5*k['mean_se'] - .14*changes.mean()
        return float(r), k

def adjacency(env):
    a = np.zeros((env.n,env.n))
    for i, ns in enumerate(env.neigh):
        a[i,ns] = 1/len(ns)
    return a

class QNet:
    """Shared MLP or learned hidden-state mean-message network."""
    def __init__(self, seed, graph=False):
        self.graph = graph
        sizes = [(10,32),(64,64),(64,5)] if graph else [(10,64),(64,64),(64,5)]
        rng = np.random.default_rng(seed)
        self.w = [rng.uniform(-np.sqrt(6/(i+j)),np.sqrt(6/(i+j)),(i,j)) for i,j in sizes]
        self.b = [np.zeros(j) for i,j in sizes]
    def params(self): return self.w+self.b
    def forward(self,x,a):
        h = np.maximum(x@self.w[0]+self.b[0],0)
        z = np.concatenate([h,a@h],axis=-1) if self.graph else h
        g = np.maximum(z@self.w[1]+self.b[1],0)
        y = g@self.w[2]+self.b[2]
        return y,(x,a,h,z,g)
    def backward(self,cache,dy):
        # dy is already the derivative of the averaged scalar loss.
        x,a,h,z,g = cache
        def wb(u,v): return np.einsum('bni,bnj->ij',u,v),v.sum((0,1))
        w2,b2=wb(g,dy)
        dg=(dy@self.w[2].T)*(g>0)
        w1,b1=wb(z,dg)
        dz=dg@self.w[1].T
        if self.graph:
            dh=dz[...,:32]+a.transpose(0,2,1)@dz[...,32:]
        else: dh=dz
        dh*=h>0
        w0,b0=wb(x,dh)
        return [w0,w1,w2,b0,b1,b2]

class Agent:
    def __init__(self,seed,kind):
        self.kind=kind; self.net=QNet(seed,kind=='graph'); self.target=copy.deepcopy(self.net)
        self.opt=base.Adam(self.net.params(),1e-3); self.rng=np.random.default_rng(seed)
        self.buffer=[]; self.updates=0
    def act(self,s,a,eps=0):
        q,_=self.net.forward(s[None],a[None]); act=q[0].argmax(-1)
        mask=self.rng.random(len(s))<eps
        act[mask]=self.rng.integers(0,5,mask.sum())
        return act
    def push(self,s,a,u,r,t):
        self.buffer.append((s.copy(),a.copy(),u.copy(),r,t.copy()))
        if len(self.buffer)>20000: self.buffer.pop(0)
    def learn(self):
        if len(self.buffer)<80:return
        ix=self.rng.integers(0,len(self.buffer),64)
        s,a,u,r,t=map(np.array,zip(*(self.buffer[i] for i in ix)))
        q,cache=self.net.forward(s,a); qt,_=self.target.forward(t,a)
        b,n=u.shape; chosen=np.take_along_axis(q,u[...,None],-1)[...,0]
        if self.kind=='vdn':
            td=chosen.sum(1)-(r+.8*qt.max(-1).sum(1))
            deriv=np.repeat((np.clip(td,-1,1)/b)[:,None],n,axis=1)
        else:
            td=chosen-(r[:,None]+.8*qt.max(-1))
            deriv=np.clip(td,-1,1)/(b*n)
        dy=np.zeros_like(q);np.put_along_axis(dy,u[...,None],deriv[...,None],-1)
        self.opt.step(self.net.params(),self.net.backward(cache,dy));self.updates+=1
        if self.updates%400==0:self.target=copy.deepcopy(self.net)
    def save(self,path):
        np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.net.params())})

SCENARIOS={
 'ring7':{},'dense7':{'isd':350.},'sparse7':{'isd':750.},
 'grid7':{'layout':'grid'},'irregular7':{'layout':'irregular'},
 'ring19':{'n_sites':19,'n_ues':543},
 'grid19':{'layout':'grid','n_sites':19,'n_ues':543},
 'irregular19':{'layout':'irregular','n_sites':19,'n_ues':543},
}

def predict(agent,s,a,env=None):
    if isinstance(agent,PPO):return agent.act(s)
    if agent=='rule':return base.rule_policy(env)
    if agent=='fixed':return np.zeros(len(s),int)
    return agent.act(s,a)

def evaluate(agent,cfg,scenario='ring7',shield=True,topos=(10,11,12,13),inits=3,steps=40):
    c=dict(cfg,**SCENARIOS[scenario]); values=[]; rejects=0; count=0
    for ts in topos:
        for init in range(inits):
            env=Env(ts,c);base._reset_eval_env(env,cfg['seed'],ts,init);a=adjacency(env)
            for _ in range(steps):
                _,_,k,reject=env.step(predict(agent,env.features(),a,env),shield=shield)
                values.append(list(k.values()));rejects+=reject;count+=1
    return dict(zip(k,np.mean(values,0)))|{'reject_pct':100*rejects/count}

def train_value(agent,cfg,episodes,curve=False,offline=False):
    if offline:
        rng=np.random.default_rng(800+cfg['seed'])
        for ep in range(75):
            env=Env(ep%6,cfg);env.rng=np.random.default_rng(3000000+100000*cfg['seed']+ep)
            s=env.reset(True);a=adjacency(env)
            for _ in range(40):
                u=base.rule_policy(env);mask=rng.random(env.n)<.2;u[mask]=rng.integers(0,5,mask.sum())
                t,r,_,_=env.step(u);agent.push(s,a,u,r,t);s=t
        for _ in range(6000):agent.learn()
    points={}
    for ep in range(episodes):
        env=Env(ep%6,cfg);env.rng=np.random.default_rng(2000000+100000*cfg['seed']+ep)
        s=env.reset(True);a=adjacency(env)
        for t in range(40):
            eps=1-.95*min(1,(ep*40+t)/(.6*episodes*40))
            u=agent.act(s,a,eps);s2,r,_,_=env.step(u)
            agent.push(s,a,u,r,s2);agent.learn();s=s2
        if curve and ep+1 in (5,10,25,50):
            points[str(ep+1)]=evaluate(agent,cfg,topos=(6,7,8,9),inits=1)
    return points

def train_ppo(agent,cfg,episodes):
    for ep in range(episodes):
        env=Env(ep%6,cfg);env.rng=np.random.default_rng(2000000+100000*cfg['seed']+ep)
        s=env.reset(True);ss=[];aa=[];ll=[];vv=[];rr=[]
        for t in range(40):
            u,lp,v=agent.sample(s);ss.append(s);aa.append(u);ll.append(lp);vv.append(v)
            s,r,_,_=env.step(u);rr.append(r)
        vv=np.array(vv);adv=np.zeros_like(vv);gae=np.zeros(env.n);_,nv=agent.distribution(s)
        for t in reversed(range(40)):
            gae=rr[t]+.8*nv-vv[t]+.8*.95*gae;adv[t]=gae;nv=vv[t]
        agent.update(np.array(ss).reshape(-1,10),np.array(aa).reshape(-1),np.array(ll).reshape(-1),adv.reshape(-1),(adv+vv).reshape(-1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);ap.add_argument('--episodes',type=int,default=250);ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    cfg=dict(base.CFG,seed=args.seed);ep=3 if args.quick else args.episodes
    result={'seed':args.seed,'episodes':ep,'config':cfg,'methods':{},'warm_start':{}}
    start=time.time()
    for name in ['dqn','graph','vdn','ppo']:
        agent=PPO(100+args.seed,cfg) if name=='ppo' else Agent(100+args.seed,name)
        print(f'seed {args.seed} {name} training',flush=True)
        if name=='ppo':train_ppo(agent,cfg,ep)
        else:train_value(agent,cfg,ep)
        scenarios=SCENARIOS if not args.quick else {'ring7':{}}
        result['methods'][name]={sc:evaluate(agent,cfg,sc,inits=1 if args.quick else 3) for sc in scenarios}
        if name!='ppo':agent.save(OUT/f'{name}_seed{args.seed}.npz')
        if name=='dqn':
            result['methods'][name]['ring7_unshielded']=evaluate(agent,cfg,shield=False)
        (OUT/f'seed{args.seed}.json').write_text(json.dumps(result,indent=2))
        print(f'seed {args.seed} {name} done {time.time()-start:.1f}s',flush=True)
    for name in ['fixed','rule']:
        result['methods'][name]={sc:evaluate(name,cfg,sc,shield=False) for sc in scenarios}
    if not args.quick:
        for warm in [False,True]:
            agent=Agent(700+args.seed,'dqn')
            result['warm_start']['warm' if warm else 'scratch']=train_value(agent,cfg,50,curve=True,offline=warm)
    result['elapsed_seconds']=time.time()-start
    (OUT/f'seed{args.seed}.json').write_text(json.dumps(result,indent=2))
    print(f'seed {args.seed} COMPLETE {time.time()-start:.1f}s',flush=True)
if __name__=='__main__':main()
