#!/usr/bin/env python3
"""Implementation checks, not validation of physical realism."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import vcc_experiments as v
from ppo_safety_scale_experiments import PPO
rng=np.random.default_rng(912);out={}
for graph in [False,True]:
 net=v.QNet(4,graph);x=rng.normal(size=(3,7,10));a=rng.uniform(size=(3,7,7));a/=a.sum(-1,keepdims=True)
 y,c=net.forward(x,a);w=rng.normal(size=y.shape)/y.size;g=net.backward(c,w);errors=[]
 for p,grad in zip(net.params(),g):
  d=rng.normal(size=p.shape);d/=np.linalg.norm(d);orig=p.copy();eps=1e-6
  p[:]=orig+eps*d;hi=(net.forward(x,a)[0]*w).sum()
  p[:]=orig-eps*d;lo=(net.forward(x,a)[0]*w).sum();p[:]=orig
  errors.append(abs((hi-lo)/(2*eps)-(grad*d).sum()))
 assert max(errors)<1e-7,errors
 out['graph_gradient_error' if graph else 'mlp_gradient_error']=max(errors)
ppo=PPO(3,dict(v.base.CFG,ppo_epochs=1,ppo_batch=8))
x=rng.normal(size=(8,10));u=rng.integers(0,5,8);pr,_=ppo.distribution(x)
old=np.log(pr[np.arange(8),u]);adv=rng.normal(size=8);adv=(adv-adv.mean())/(adv.std()+1e-8)
captured={};ppo.actor_opt.step=lambda p,g:captured.update(actor=g);ppo.critic_opt.step=lambda p,g:captured.update(critic=g)
ret=rng.normal(size=8);ppo.update(x,u,old,adv,ret)
def objective():
 pr,val=ppo.distribution(x);ratio=pr[np.arange(8),u]/np.exp(old);entropy=-(pr*np.log(pr)).sum(1)
 return -np.mean(np.minimum(ratio*adv,np.clip(ratio,.8,1.2)*adv))-.01*entropy.mean(),.25*np.mean((val-ret)**2)
errors=[]
for j,net in enumerate([ppo.actor,ppo.critic]):
 for p,g in zip(net.params(),captured['actor' if j==0 else 'critic']):
  d=rng.normal(size=p.shape);d/=np.linalg.norm(d);orig=p.copy();eps=1e-6
  p[:]=orig+eps*d;hi=objective()[j];p[:]=orig-eps*d;lo=objective()[j];p[:]=orig
  errors.append(abs((hi-lo)/(2*eps)-(g*d).sum()))
assert max(errors)<1e-7,max(errors)
out['ppo_gradient_error']=max(errors)
cfg=dict(v.base.CFG,dynamic=False);env=v.Env(10,cfg);sinr=[];rx=[]
for u in range(cfg['n_ues']):
 powers=[]
 for i in range(env.n):
  dx=env.sites[i,0]-env.ue[u,0];dy=env.sites[i,1]-env.ue[u,1];r=math.sqrt(dx*dx+dy*dy)+1;h=23.5;d=math.sqrt(r*r+h*h)
  pl=28+22*math.log10(d)+20*math.log10(3.5)
  att=min(12*((math.degrees(math.atan2(h,r))-env.tilt[i])/10)**2,20)
  powers.append(env.power[i]+8-att-pl-env.shadow[i,u])
 s=max(range(env.n),key=lambda i:powers[i]);p=[10**(x/10) for x in powers];noise=10**((-174+10*math.log10(20e6)+9)/10)
 sinr.append(p[s]/(sum(x for i,x in enumerate(p) if i!=s)+noise));rx.append(powers[s])
err=float(np.max(np.abs(np.array(sinr)-env.sinr)));assert err<1e-9,err;assert np.allclose(rx,env.rsrp_serv)
out['scalar_radio_max_sinr_error']=err
before=env.kpis()['coverage'];_,_,k,_=env.step(np.zeros(env.n,int));assert k['coverage']==before
for _ in range(200):
 before=env.kpis()['coverage'];_,_,k,_=env.step(rng.integers(0,5,env.n));assert k['coverage']>=before-.02-1e-12
out.update(frozen_noop='PASS',oracle_floor_200_actions='PASS')
(v.OUT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
