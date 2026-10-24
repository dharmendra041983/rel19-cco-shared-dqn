#!/usr/bin/env python3
"""Independent ns-3.44 propagation kernels in the existing control harness.
No claim of ns-3 packet/NR-stack validation. Native LOS/NLOS pathloss has
shadowing disabled; the paired Python log-distance condition does too.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,ctypes,json
from pathlib import Path
import numpy as np
import vcc_experiments as v

class Kernel:
 def __init__(self,library=None):
  self.lib=ctypes.CDLL(str(library or Path(__file__).parent/'validation/ns3_bridge.so'))
  self.lib.bridge_create.restype=ctypes.c_void_p
  self.lib.bridge_delete.argtypes=[ctypes.c_void_p]
  self.lib.bridge_losses.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
  self.handle=self.lib.bridge_create()
 def losses(self,sites,users,mode):
  a=np.ascontiguousarray(sites,dtype=np.float64);b=np.ascontiguousarray(users,dtype=np.float64);out=np.zeros((len(a),len(b)),dtype=np.float64)
  self.lib.bridge_losses(self.handle,a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(a),b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(b),mode,out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  return out

KERNEL=None
class KernelEnv(v.Env):
 def _compute(self):
  cfg=self.cfg;delta=self.sites[:,None,:]-self.ue[None,:,:];horizontal=np.linalg.norm(delta,axis=-1)
  d2=horizontal+1;dh=cfg['bs_height']-cfg['ue_height'];d3=np.sqrt(d2*d2+dh*dh)
  mode=cfg.get('kernel','python')
  if mode=='python':pl=28+22*np.log10(d3)+20*np.log10(3.5)
  else:
   if hasattr(self,'_cached_users') and np.array_equal(self.ue,self._cached_users):pl=self._cached_pl
   else:
    pl=KERNEL.losses(self.sites,self.ue,{'los':0,'nlos':1,'logdistance':2}[mode])
    self._cached_users=self.ue.copy();self._cached_pl=pl
  elev=np.degrees(np.arctan2(dh,d2));att=np.minimum(12*((elev-self.tilt[:,None])/10)**2,20)
  self.rsrp=self.power[:,None]+8-att-pl-self.shadow
  self.serv=self.rsrp.argmax(0);p=10**(self.rsrp/10);j=np.arange(len(self.ue));signal=p[self.serv,j]
  self.sinr=signal/(p.sum(0)+self.noise_mw-signal);self.sinr_db=10*np.log10(self.sinr);self.rsrp_serv=self.rsrp[self.serv,j]
  self.load=np.bincount(self.serv,minlength=self.n);self.rate=cfg['bw_hz']/np.maximum(self.load[self.serv],1)*np.log2(1+self.sinr)
  pi=p.copy();pi[self.serv,j]=-1;self.top_int=pi.argmax(0)


def evaluate(agent,seed,sc,kernel):
 cfg=dict(v.base.CFG,seed=seed,shadow_sigma_db=0.,kernel=kernel,**v.SCENARIOS[sc]);ks=[];rejects=0
 for ts in (10,11,12,13):
  for init in range(3):
   env=KernelEnv(ts,cfg);v.base._reset_eval_env(env,seed,ts,init);a=v.adjacency(env)
   for _ in range(40):
    u=v.predict(agent,env.features(),a,env);_,_,k,rejected=env.step(u,shield=not isinstance(agent,str));ks.append(list(k.values()));rejects+=rejected
 return dict(zip(k,np.mean(ks,0)))|{'reject_pct':100*rejects/len(ks)}

def checks(seed):
 cfg=dict(v.base.CFG,seed=seed,shadow_sigma_db=0.,dynamic=False);env=v.Env(10,cfg)
 pl=KERNEL.losses(env.sites,env.ue,2);d2=np.linalg.norm(env.sites[:,None,:]-env.ue[None,:,:],axis=-1)+1
 reference=28+22*np.log10(np.sqrt(d2*d2+23.5**2))+20*np.log10(3.5)
 err=float(np.max(np.abs(pl-reference)));assert err<1e-9,err
 # Distances chosen within the UMa range; beyond-breakpoint LOS is intentional.
 sites=np.array([[0.,0.]]);ue=np.array([[d,0.] for d in [10.,50.,100.,500.,1000.,2000.]])
 los=KERNEL.losses(sites,ue,0)[0];nlos=KERNEL.losses(sites,ue,1)[0]
 assert np.all(nlos>=los-1e-10)
 return {'logdistance_max_abs_db_error':err,'distance_sweep_m':[10,50,100,500,1000,2000],'uma_los_loss_db':los.tolist(),'uma_nlos_loss_db':nlos.tolist()}

def main():
 global KERNEL
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();seed=args.seed;KERNEL=Kernel()
 result={'seed':seed,'ns3_version':'3.44.post0','scope':'ns-3 native propagation kernel; custom association/interference/scheduling/control; no packet stack','shadowing_db':0,'checks':checks(seed),'conditions':{}}
 for sc in ['ring7','ring19']:
  result['conditions'][sc]={}
  for mode in ['python','los','nlos']:
   result['conditions'][sc][mode]={}
   for name in ['fixed','rule','dqn','vdn','vdn_mean']:
    if name in ['fixed','rule']:agent=name
    else:
     agent=v.Agent(100+seed,'dqn' if name=='dqn' else 'vdn');w=np.load(v.OUT/f'{name}_seed{seed}.npz')
     for i,p in enumerate(agent.net.params()):p[:]=w[f'p{i}']
    result['conditions'][sc][mode][name]=evaluate(agent,seed,sc,mode)
   print('ns3',seed,sc,mode,'complete',flush=True)
 (v.OUT/f'ns3_seed{seed}.json').write_text(json.dumps(result,indent=2));print('ns3',seed,'COMPLETE',flush=True)
if __name__=='__main__':main()
