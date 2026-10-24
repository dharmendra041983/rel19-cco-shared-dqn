#!/usr/bin/env python3
"""Numerical gates for the added training loss and native-kernel adapter."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import vcc_experiments as v
import vcc_ns3_validation as native
rng=np.random.default_rng(110)
net=v.QNet(3);x=rng.normal(size=(4,7,10));a=rng.random((4,7,7));a/=a.sum(-1,keepdims=True)
u=rng.integers(0,5,(4,7));target=rng.normal(size=4)
q,cache=net.forward(x,a);td=np.take_along_axis(q,u[...,None],-1)[...,0].mean(1)-target
huber_gradient=np.clip(td,-1,1);dy=np.zeros_like(q);np.put_along_axis(dy,u[...,None],np.repeat((huber_gradient/28)[:,None],7,1)[...,None],-1)
grad=net.backward(cache,dy)
def objective():
 q,_=net.forward(x,a);r=np.take_along_axis(q,u[...,None],-1)[...,0].mean(1)-target
 return np.where(np.abs(r)<=1,.5*r*r,np.abs(r)-.5).mean()
errors=[]
for p,g in zip(net.params(),grad):
 d=rng.normal(size=p.shape);d/=np.linalg.norm(d);saved=p.copy();eps=1e-6
 p[:]=saved+eps*d;hi=objective();p[:]=saved-eps*d;lo=objective();p[:]=saved
 errors.append(abs((hi-lo)/(2*eps)-(g*d).sum()))
assert max(errors)<1e-7,max(errors)
native.KERNEL=native.Kernel();checks=native.checks(0)
for mode in ['los','nlos']:
 c=dict(v.base.CFG,seed=0,shadow_sigma_db=0.,kernel=mode)
 env=native.KernelEnv(10,c)
 cached=env.sinr.copy();del env._cached_users;env._compute();assert np.array_equal(cached,env.sinr)
R=ROOT/'vcc_results'
for seed in range(5):
 old=json.loads((R/f'safety_seed{seed}.json').read_text());new=json.loads((R/f'margin_seed{seed}.json').read_text())
 for sc in new['scenarios']:
  for k in ['coverage','violation_pct','reject_pct']:
   assert abs(new['scenarios'][sc]['mixed_q95'][k]-old['scenarios'][sc]['surrogate'][k])<1e-10
 n=json.loads((R/f'normalization_seed{seed}.json').read_text());assert n['frozen_sum_invariance']['normalization_disagreements']==0
build=json.loads((R/'ns3_build.json').read_text());assert build['adapter_source_sha256']==hashlib.sha256((ROOT/'validation/ns3_bridge.cc').read_bytes()).hexdigest()
result={'mean_vdn_gradient_max_error':max(errors),'ns3_logdistance_error_db':checks['logdistance_max_abs_db_error'],'native_cache_equivalence':'PASS','previous_q95_gate_reproduction':'PASS','frozen_normalization_action_invariance':'PASS','adapter_source_hash':'PASS'}
(R/'ablation_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
